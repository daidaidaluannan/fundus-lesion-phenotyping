#!/usr/bin/env python3
"""
FastAPI demo for counterfactual / latent-transfer visualization.

Features:
- Upload single image; models are loaded from preset env paths (AAE_C, AOT-GAN, AAE_S).
- Runs AOT-GAN masking + pseudo-healthy reconstruction, AAE_S latent extraction.
- Produces key visualizations: mask, pseudo image, latent overlays (mean/max),
  optional latent adjustment panel (scale steps), optional Top-K latent transfer result.

Notes:
- Designed for local testing; batch size = 1; models cached per color mode.
- Uses existing train_demo utilities; ensure PYTHONPATH includes release_bundle root when running.
"""
import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms, utils as vutils
from torchvision.transforms import functional as TF
import traceback
import json

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
# Ensure train_demo modules (e.g., AAE_S import style) are importable
train_demo_dir = REPO_ROOT / "train_demo"
if str(train_demo_dir) not in sys.path:
    sys.path.append(str(train_demo_dir))

from web import config  # noqa: E402
from train_demo import aotgan_1123 as aot  # noqa: E402
from train_demo.visualize_multimap_single_1113 import (  # noqa: E402
    ModelConfig,
    build_multimap_model,
    create_latent_overlay,
    build_fov_mask,
    adjust_latent_and_decode,
    normalize_tensor,
)
from train_demo.change_latent import (  # noqa: E402
    compute_spatial_diff_map,
    pick_top_spatial_positions,
    replace_latent_positions,
)
from transforms.synthetic import GenerateMasks  # noqa: E402
from model_zoo.phanes import AnomalyMap  # noqa: E402
from train_demo.counterfactual_pipeline import latent_heatmap_7x7  # noqa: E402

SAMPLE_DIR = Path(__file__).parent / "exp_pic"


app = FastAPI(title="PHANES Counterfactual Web Demo", version="0.1")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def load_models(color_mode: str) -> Dict[str, Any]:
    paths = config.MODEL_PATHS
    if (not paths) or any(k not in paths for k in ("aae_c", "aot_gan", "aae_s")):
        # Try loading from persisted file (for reload process)
        cfg_file = Path(__file__).parent / "model_paths.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                loaded = json.load(f)
            if "paths" in loaded:
                config.MODEL_PATHS.update(loaded["paths"])
            if "ui" in loaded:
                config.UI_SETTINGS.update(loaded["ui"])
            paths = config.MODEL_PATHS
    if not paths or any(k not in paths for k in ("aae_c", "aot_gan", "aae_s")):
        raise RuntimeError("Missing model paths. Please start server via run_server.py with model arguments.")
    default_device = paths.get("device", "cuda")
    key = f"{color_mode}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    device = torch.device(default_device if torch.cuda.is_available() or default_device == "cpu" else "cpu")
    ret_aae = aot.load_ret_aae(Path(paths["aae_c"]), device)
    netG, _ = aot.build_aotgan_networks(color_mode, device)
    ckpt = torch.load(paths["aot_gan"], map_location=device)
    state = ckpt.get("netG", ckpt)
    netG.load_state_dict(state)
    netG.eval()
    multimap_model, multimap_cfg = build_multimap_model(
        ModelConfig(ckpt_path=paths["aae_s"], config_path=None),
        device=device,
    )
    multimap_model.eval()
    MODEL_CACHE[key] = {
        "device": device,
        "ret_aae": ret_aae,
        "netG": netG,
        "multimap": multimap_model,
        "multimap_cfg": multimap_cfg,
    }
    return MODEL_CACHE[key]


def run_pipeline(image: Image.Image,
                 color_mode: str,
                 mask_threshold: float,
                 aot_img_size: int,
                 viz_img_size: int,
                 top_percent: float,
                 latent_scale_step: float,
                 latent_scale_max: float,
                 do_topk_transfer: bool = True,
                 do_latent_adjust: bool = True) -> Dict[str, Any]:
    models = load_models(color_mode)
    device = models["device"]

    tfm = transforms.Compose([
        transforms.Resize((aot_img_size, aot_img_size)),
        transforms.ToTensor(),
    ])
    img_tensor = tfm(image.convert("RGB")).unsqueeze(0).to(device)

    # AOT-GAN stage
    ret_aae = models["ret_aae"]
    netG = models["netG"]
    ano = AnomalyMap(device=device)
    mask_generator = GenerateMasks(min_size=20, max_size=40)

    coarse_rgb, _ = ret_aae(img_tensor)
    x_gray = TF.rgb_to_grayscale(img_tensor)
    coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
    mask = aot.generate_mask(
        ano,
        mask_generator,
        x_gray,
        coarse_gray,
        mask_threshold=mask_threshold,
        add_synthetic=False)

    if color_mode == "color":
        mask_rgb = mask.repeat(1, 3, 1, 1)
        transformed = (img_tensor * (1 - mask_rgb)) + mask_rgb
    else:
        mask_rgb = mask
        transformed = (x_gray * (1 - mask)) + mask

    pred = netG(transformed, mask)
    if color_mode == "gray":
        pred = pred.repeat(1, 3, 1, 1)
    pred = pred.clamp(0.0, 1.0)

    # Multimap stage
    multimap = models["multimap"]
    orig_viz = F.interpolate(img_tensor, size=(viz_img_size, viz_img_size), mode="bilinear", align_corners=False)
    pseudo_viz = F.interpolate(pred, size=(viz_img_size, viz_img_size), mode="bilinear", align_corners=False)
    recon_orig, latent_orig = multimap(orig_viz)
    recon_pseudo, latent_pseudo = multimap(pseudo_viz)
    recon_orig = recon_orig.clamp(0, 1)
    recon_pseudo = recon_pseudo.clamp(0, 1)
    if latent_orig.shape != latent_pseudo.shape:
        latent_pseudo = F.interpolate(latent_pseudo, size=latent_orig.shape[-2:], mode="bilinear", align_corners=False)

    fov_mask = build_fov_mask(orig_viz.squeeze(0), threshold=0.05, erode_kernel=7)
    mask_latent = F.interpolate(
        fov_mask.unsqueeze(0).unsqueeze(0),
        size=latent_orig.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    overlays = {}
    for reduction in ["mean", "max"]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            create_latent_overlay(
                original=orig_viz.squeeze(0),
                multimap_latent=latent_orig.squeeze(0),
                baseline_latent=latent_pseudo.squeeze(0),
                out_path=tmp_path,
                alpha=0.5,
                threshold=0.0,
                gamma=1.0,
                reduction=reduction,
                fusion_mode=None,
                fusion_weight=0.5,
                top_pct=None,
                mask=mask_latent,
            )
            with open(tmp_path, "rb") as f:
                overlays[reduction] = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    adjust_panel_b64 = None
    if do_latent_adjust:
        panel_imgs = [orig_viz.squeeze(0), pseudo_viz.squeeze(0)]
        panel_labels = ["Original", "Pseudo"]
        scales = build_scale_sequence(latent_scale_step, latent_scale_max)
        for scale in scales:
            adjust_map = (latent_pseudo - latent_orig) * scale
            recon_adj = adjust_latent_and_decode(multimap, latent_spatial=latent_orig, adjust_map=adjust_map).squeeze(0).clamp(0, 1)
            panel_imgs.append(recon_adj)
            panel_labels.append(f"Adjust x{scale:g}")
        panel = stack_h(panel_imgs, panel_labels)
        adjust_panel_b64 = pil_to_base64(panel)

    heatmap_b64 = None
    try:
        with torch.no_grad():
            diff_map = compute_spatial_diff_map(latent_orig, latent_pseudo, reduction="mean")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            latent_heatmap_7x7(diff_map, Path(tmp_path))
            with open(tmp_path, "rb") as f:
                heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"[heatmap] generation failed: {e}")
        heatmap_b64 = None

    topk_b64 = None

    # bundle zip for download
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        def add(name, b64):
            if b64:
                zf.writestr(name, base64.b64decode(b64))
        add("input.png", pil_to_base64(tensor_to_pil(img_tensor)))
        add("mask.png", pil_to_base64(tensor_to_pil(mask_rgb)))
        add("pseudo.png", pil_to_base64(tensor_to_pil(pred)))
        add("coarse.png", pil_to_base64(tensor_to_pil(coarse_rgb)))
        add("overlay_mean.png", overlays.get("mean"))
        add("overlay_max.png", overlays.get("max"))
        add("topk_transfer.png", topk_b64)
        add("latent_adjust_panel.png", adjust_panel_b64)
        add("latent_heatmap_7x7.png", heatmap_b64)
    zip_buf.seek(0)
    zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

    return {
        "input": pil_to_base64(tensor_to_pil(img_tensor)),
        "mask": pil_to_base64(tensor_to_pil(mask_rgb)),
        "pseudo": pil_to_base64(tensor_to_pil(pred)),
        "coarse": pil_to_base64(tensor_to_pil(coarse_rgb)),
        "overlay_mean": overlays.get("mean"),
        "overlay_max": overlays.get("max"),
        "adjust_panel": adjust_panel_b64,
        "latent_heatmap_7x7": heatmap_b64,
        "topk_transfer": topk_b64,
        "zip": zip_b64,
    }


def build_scale_sequence(step: float, max_scale: float = 1.0) -> List[float]:
    if step <= 0:
        return [0.0, max_scale]
    values, cur = [], 0.0
    eps = 1e-6
    while cur < max_scale - eps:
        values.append(round(cur, 6))
        cur += step
    if not values or abs(values[-1] - max_scale) > eps:
        values.append(float(max_scale))
    return values


def stack_h(images: List[torch.Tensor], labels: List[str]) -> Image.Image:
    pil_imgs = [tensor_to_pil(t) for t in images]
    widths = [p.width for p in pil_imgs]
    heights = [p.height for p in pil_imgs]
    total_w = sum(widths)
    max_h = max(heights)
    panel = Image.new("RGB", (total_w, max_h))
    try:
        from PIL import ImageDraw, ImageFont
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except Exception:
        from PIL import ImageDraw, ImageFont
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    x = 0
    for img, label in zip(pil_imgs, labels):
        panel.paste(img, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        bg = [x + 6, 6, x + 6 + tw + pad * 2, 6 + th + pad * 2]
        draw.rectangle(bg, fill=(0, 0, 0))
        draw.text((x + 6 + pad, 6 + pad), label, fill=(255, 255, 255), font=font)
        x += img.width
    return panel


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = static_dir / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def favicon():
    icon_path = static_dir / "logo.png"
    if icon_path.exists():
        return FileResponse(icon_path)
    return JSONResponse(status_code=404, content={"error": "favicon not found"})


@app.get("/config")
def get_config():
    return {
        "layout": config.UI_SETTINGS.get("layout", "vertical")
    }


@app.get("/examples")
def list_examples():
    items = []
    if SAMPLE_DIR.exists():
        for p in sorted(SAMPLE_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                items.append(p.name)
    return {"examples": items}


def _as_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    return val in ("1", "true", "yes", "on")


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(None),
    sample_name: Optional[str] = Form(None),
    color_mode: str = Form("gray"),
    mask_threshold: float = Form(0.153),
    aot_img_size: int = Form(224),
    viz_img_size: int = Form(512),
    top_percent: float = Form(10.0),
    latent_scale_step: float = Form(0.25),
    latent_scale_max: float = Form(1.0),
    do_topk_transfer: Optional[str] = Form("true"),
    do_latent_adjust: Optional[str] = Form("true"),
):
    pil_img = None
    # Prefer uploaded file if valid
    if image is not None and getattr(image, "filename", None):
        data = await image.read()
        if data:
            try:
                pil_img = Image.open(io.BytesIO(data))
            except Exception:
                pil_img = None
    # Fallback to sample
    if pil_img is None and sample_name:
        sample_path = (SAMPLE_DIR / sample_name).resolve()
        try:
            sample_path.relative_to(SAMPLE_DIR.resolve())
        except Exception:
            return JSONResponse({"error": "Invalid sample name"}, status_code=400)
        if not sample_path.is_file():
            return JSONResponse({"error": "Sample not found"}, status_code=404)
        pil_img = Image.open(sample_path)

    if pil_img is None:
        return JSONResponse({"error": "No image uploaded"}, status_code=400)
    try:
        result = run_pipeline(
            pil_img,
            color_mode=color_mode,
            mask_threshold=mask_threshold,
            aot_img_size=aot_img_size,
            viz_img_size=viz_img_size,
            top_percent=top_percent,
            latent_scale_step=latent_scale_step,
            latent_scale_max=latent_scale_max,
            do_topk_transfer=_as_bool(do_topk_transfer, True),
            do_latent_adjust=_as_bool(do_latent_adjust, True),
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
