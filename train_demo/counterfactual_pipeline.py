#!/usr/bin/env python3
"""
Counterfactual pipeline:
1. Use pretrained AAE_C + AOT-GAN (from aotgan_1123) to generate automatic mask
   and pseudo-healthy reconstruction for a given image.
2. Feed both original and pseudo-healthy images into trained AAE_S (multimap)
   and visualize latent differences / optional latent adjustment heatmaps using
   helpers from visualize_multimap_single_1113.py.
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

repo_root = Path(__file__).resolve().parent.parent
import sys
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from model_zoo.phanes import AnomalyMap  # noqa: E402
from transforms.synthetic import GenerateMasks  # noqa: E402
from train_demo import aotgan_1123 as aot  # noqa: E402
from train_demo.visualize_multimap_single_1113 import (  # noqa: E402
    VisualizationOutputs,
    build_multimap_model,
    ModelConfig,
    create_latent_overlay,
    build_fov_mask,
    adjust_latent_and_decode,
    normalize_tensor,
    save_tensor_image,
    save_latent_grid,
    ensure_dir,
)


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)  # B,C,H,W
    return tensor


@torch.no_grad()
def run_aotgan_inference(
        image_tensor: torch.Tensor,
        aae_ckpt: Path,
        aot_ckpt: Path,
        mask_threshold: float,
        color_mode: str,
        device: torch.device) -> dict:
    """
    Given a single image tensor (1,C,H,W), generate coarse reconstruction,
    automatic mask, and pseudo-healthy reconstruction using pretrained AAE_C
    and AOT-GAN.
    """
    ret_aae = aot.load_ret_aae(aae_ckpt, device)
    netG, _ = aot.build_aotgan_networks(color_mode, device)
    ckpt = torch.load(aot_ckpt, map_location=device)
    state = ckpt.get("netG", ckpt)
    netG.load_state_dict(state)
    netG.eval()

    ano = AnomalyMap()
    mask_generator = GenerateMasks(min_size=20, max_size=40)

    rgb = image_tensor.to(device)
    coarse_rgb, _ = ret_aae(rgb)
    x_gray = TF.rgb_to_grayscale(rgb)
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
        transformed = (rgb * (1 - mask_rgb)) + mask_rgb
    else:
        mask_rgb = mask
        transformed = (x_gray * (1 - mask)) + mask

    pred = netG(transformed, mask)
    if color_mode == "gray":
        pred = pred.repeat(1, 3, 1, 1)
    pred = pred.clamp(0.0, 1.0)

    return {
        "input": rgb,
        "coarse": coarse_rgb.clamp(0, 1),
        "mask": mask,
        "mask_rgb": mask_rgb,
        "pseudo": pred,
    }


@torch.no_grad()
def run_multimap(model, image: torch.Tensor, out_dir: Path, tag: str) -> VisualizationOutputs:
    ensure_dir(out_dir)
    recon, latent = model(image)
    recon = recon.clamp(0, 1)
    diff = torch.abs(image - recon)

    save_tensor_image(image.squeeze(0), out_dir / f"{tag}_input.png")
    save_tensor_image(recon.squeeze(0), out_dir / f"{tag}_recon.png")
    save_tensor_image(normalize_tensor(diff).squeeze(0), out_dir / f"{tag}_diff.png")

    latent_spatial = latent if latent.ndim == 4 else None
    if latent_spatial is not None:
        save_latent_grid(latent_spatial.mean(dim=1, keepdim=True), out_dir / f"{tag}_latent_mean.png")
        save_latent_grid(latent_spatial.max(dim=1, keepdim=True).values, out_dir / f"{tag}_latent_max.png")

    return VisualizationOutputs(
        input=image.squeeze(0),
        recon=recon.squeeze(0),
        diff=diff.squeeze(0),
        latent_spatial=latent_spatial,
        latent_global=None,
        extra={},
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def save_adjustment_panel(tensors, labels, out_path: Path) -> None:
    if len(tensors) != len(labels):
        raise ValueError("tensors and labels must match in length")
    pil_imgs = [tensor_to_pil(t) for t in tensors]
    widths = [img.width for img in pil_imgs]
    heights = [img.height for img in pil_imgs]
    total_width = sum(widths)
    max_height = max(heights)
    panel = Image.new("RGB", (total_width, max_height))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    x_offset = 0
    for img, label in zip(pil_imgs, labels):
        panel.paste(img, (x_offset, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 4
        bg_coords = [
            x_offset + 6,
            6,
            x_offset + 6 + text_w + pad * 2,
            6 + text_h + pad * 2,
        ]
        draw.rectangle(bg_coords, fill=(0, 0, 0))
        draw.text((x_offset + 6 + pad, 6 + pad), label, fill=(255, 255, 255), font=font)
        x_offset += img.width
    panel.save(out_path)


def combine_overlay_variants(image_paths, labels, out_path: Path) -> None:
    valid = []
    for path, label in zip(image_paths, labels):
        if path.exists():
            valid.append((Image.open(path).convert("RGB"), label))
    if not valid:
        return
    widths = [img.width for img, _ in valid]
    heights = [img.height for img, _ in valid]
    panel = Image.new("RGB", (sum(widths), max(heights)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=24)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    offset = 0
    for img, label in valid:
        panel.paste(img, (offset, 0))
        if label:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 6
            rect = [offset + 8, 8, offset + 8 + tw + pad * 2, 8 + th + pad * 2]
            draw.rectangle(rect, fill=(0, 0, 0))
            draw.text((offset + 8 + pad, 8 + pad), label, fill=(255, 255, 255), font=font)
        offset += img.width
    panel.save(out_path)


def save_channel_diff_grid(original: torch.Tensor,
                           baseline: torch.Tensor,
                           base_image: torch.Tensor,
                           overlay_fn,
                           out_path: Path,
                           cols: int = 6,
                           tile_size: int = 224) -> None:
    if original is None or baseline is None:
        return
    tensor = original.detach().cpu()
    baseline = baseline.detach().cpu()
    if tensor.shape != baseline.shape:
        return
    base_img = base_image.detach().cpu()
    if base_img.ndim == 4:
        base_img = base_img.squeeze(0)
    if base_img.shape[0] == 1:
        base_img = base_img.repeat(3, 1, 1)
    base_img_resized = F.interpolate(
        base_img.unsqueeze(0),
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    b, c, h, w = tensor.shape
    rows = math.ceil(c / cols)
    panel = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(0, 0, 0))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
    for idx in range(c):
        ch_orig = tensor[:, idx:idx+1, :, :]
        ch_base = baseline[:, idx:idx+1, :, :]
        ch_orig_res = F.interpolate(
            ch_orig,
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        ch_base_res = F.interpolate(
            ch_base,
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tile_path = out_path.parent / f"_tmp_channel_{idx}.png"
        overlay_fn(
            original=base_img_resized,
            multimap_latent=ch_orig_res,
            baseline_latent=ch_base_res,
            out_path=tile_path,
            alpha=0.5,
            threshold=0.0,
            gamma=1.0,
            reduction="mean",
            fusion_mode=None,
            fusion_weight=0.5,
            top_pct=None,
            mask=None,
        )
        tile = Image.open(tile_path).resize((tile_size, tile_size))
        tile_path.unlink(missing_ok=True)
        draw = ImageDraw.Draw(tile)
        label = f"Ch {idx}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        rect = [6, 6, 6 + tw + pad * 2, 6 + th + pad * 2]
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((6 + pad, 6 + pad), label, fill=(255, 255, 255), font=font)
        r, c_idx = divmod(idx, cols)
        panel.paste(tile, (c_idx * tile_size, r * tile_size))
    panel.save(out_path)


def latent_heatmap_7x7(diff_map: torch.Tensor, out_path: Path, cmap_name: str = "coolwarm") -> None:
    """
    将 latent 差异图聚合为 7x7 网格并可视化：蓝色接近正常，红色越异常，格子内显示数值。
    """
    if diff_map.ndim == 4:
        diff_map = diff_map.mean(dim=1)  # (B,H,W)
    if diff_map.ndim != 3:
        raise ValueError("diff_map should be (B,H,W)")

    diff = diff_map.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    diff_grid = F.adaptive_avg_pool2d(diff, output_size=(7, 7)).squeeze().detach().cpu().numpy()
    vmin, vmax = float(diff_grid.min()), float(diff_grid.max())
    if vmax - vmin < 1e-8:
        norm_grid = np.zeros_like(diff_grid)
    else:
        norm_grid = (diff_grid - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(norm_grid)[..., :3] * 255).astype(np.uint8)
    # upscale for readability
    cell_size = 70  # px per cell
    img = Image.fromarray(colored).resize((cell_size * 7, cell_size * 7), resample=Image.NEAREST)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
    h, w = norm_grid.shape
    for r in range(h):
        for c in range(w):
            val = diff_grid[r, c]
            text = f"{val:.3f}"
            x = c * cell_size + cell_size / 2
            y = r * cell_size + cell_size / 2
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw / 2, y - th / 2), text, font=font, fill=(0, 0, 0),
                      stroke_width=2, stroke_fill=(255, 255, 255))
    # grid lines
    for i in range(8):
        draw.line([(0, i * cell_size), (cell_size * 7, i * cell_size)], fill=(255, 255, 255), width=2)
        draw.line([(i * cell_size, 0), (i * cell_size, cell_size * 7)], fill=(255, 255, 255), width=2)
    img.save(out_path)


#def build_scale_sequence(step: float):
#    if step <= 0:
#        return [1.0]
#    values = []
#    cur = 0.0
#    eps = 1e-6
#    while cur < 1.0 - eps:
#        values.append(round(cur, 6))
#        cur += step
#    if not values or abs(values[-1] - 1.0) > eps:
#        values.append(1.0)
#    return values
def build_scale_sequence(step: float, max_scale: float = 1.0):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Counterfactual pipeline combining AOT-GAN and AAE_S visualizations.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--aae-c-ckpt", required=True, help="Checkpoint for pretrained AAE_C (Ret_AAE).")
    parser.add_argument("--aot-ckpt", required=True, help="Checkpoint for trained AOT-GAN (expects netG).")
    parser.add_argument("--mask-threshold", type=float, default=0.153, help="Masking threshold used for AOT-GAN.")
    parser.add_argument("--color-mode", choices=["gray", "color"], default="gray", help="Whether the AOT-GAN was trained on grayscale or RGB.")
    parser.add_argument("--aot-img-size", type=int, default=224, help="Resize for AOT-GAN stage.")
    parser.add_argument("--viz-img-size", type=int, default=512, help="Resize for multimap visualization.")

    parser.add_argument("--multimap-ckpt", required=True, help="Checkpoint for AAE_S / multimap model.")
    parser.add_argument("--multimap-config", default=None, help="JSON config for multimap model.")
    parser.add_argument("--latent-scale", type=float, default=1.0,
                        help="Step size for latent pushes; e.g. 0.2 yields 0,0.2,...,1.0 adjustments.")
    parser.add_argument("--latent-max-scale", type=float, default=1.0, help="latent-max-scale")
    parser.add_argument("--adjust-latent", action="store_true", help="Perform latent adjustment using pseudo image latent.")
    parser.add_argument("--fov-threshold", type=float, default=0.05,
                        help="Threshold used in build_fov_mask to separate foreground.")
    parser.add_argument("--fov-erode", type=int, default=7,
                        help="Odd kernel size for mask erosion in build_fov_mask.")
    parser.add_argument("--disable-fov-mask", action="store_true",
                        help="Disable FOV masking when creating latent overlays.")

    parser.add_argument("--out-dir", required=True, help="Output directory for storing intermediate and final results.")
    parser.add_argument("--device", default="cuda", help="Computation device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("[pipeline] CUDA unavailable, falling back to CPU.")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1) Load image for AOT-GAN stage
    img_for_aot = load_image(Path(args.image), args.aot_img_size, device)
    aot_outputs = run_aotgan_inference(
        image_tensor=img_for_aot,
        aae_ckpt=Path(args.aae_c_ckpt),
        aot_ckpt=Path(args.aot_ckpt),
        mask_threshold=args.mask_threshold,
        color_mode=args.color_mode,
        device=device,
    )

    save_tensor_image(aot_outputs["mask"].squeeze(0), out_dir / "auto_mask.png")
    save_tensor_image(aot_outputs["pseudo"].squeeze(0), out_dir / "pseudo_healthy.png")
    save_tensor_image(aot_outputs["coarse"].squeeze(0), out_dir / "coarse_recon.png")

    # 2) Resize images for multimap visualization (AAE_S)
    orig_viz = F.interpolate(
        aot_outputs["input"],
        size=(args.viz_img_size, args.viz_img_size),
        mode="bilinear",
        align_corners=False,
    )
    pseudo_viz = F.interpolate(
        aot_outputs["pseudo"],
        size=(args.viz_img_size, args.viz_img_size),
        mode="bilinear",
        align_corners=False,
    )

    multimap_model, multimap_cfg = build_multimap_model(
        ModelConfig(ckpt_path=args.multimap_ckpt, config_path=args.multimap_config),
        device=device)
    multimap_model.eval()

    orig_out = run_multimap(multimap_model, orig_viz.to(device), out_dir, "original")
    pseudo_out = run_multimap(multimap_model, pseudo_viz.to(device), out_dir, "pseudo")

    # 3) Latent overlays and adjustments
    if orig_out.latent_spatial is not None and pseudo_out.latent_spatial is not None:
        latent_orig_full = orig_out.latent_spatial
        latent_pseudo_full = pseudo_out.latent_spatial
        if latent_orig_full.shape != latent_pseudo_full.shape:
            latent_pseudo_full = F.interpolate(
                latent_pseudo_full,
                size=latent_orig_full.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        #latent_orig = latent_orig_full.mean(dim=1)
        #latent_pseudo = latent_pseudo_full.mean(dim=1)
        latent_orig = latent_orig_full.squeeze(0)
        latent_pseudo = latent_pseudo_full.squeeze(0)

        fov_mask = build_fov_mask(orig_out.input, threshold=args.fov_threshold, erode_kernel=args.fov_erode)
        mask_latent = F.interpolate(
            fov_mask.unsqueeze(0).unsqueeze(0),
            size=latent_orig.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        overlay_paths = []
        for reduction in ["mean", "max"]:
            overlay_path = out_dir / f"latent_overlay_original_vs_pseudo_{reduction}.png"
            create_latent_overlay(
                original=orig_out.input,
                multimap_latent=latent_orig,
                baseline_latent=latent_pseudo,
                out_path=overlay_path,
                alpha=0.5,
                threshold=0.0,
                gamma=1.0,
                reduction=reduction,
                fusion_mode=None,
                fusion_weight=0.5,
                top_pct=None,
                mask=mask_latent,
            )
            overlay_paths.append((overlay_path, reduction.title()))
        combine_overlay_variants(
            [p for p, _ in overlay_paths],
            [label for _, label in overlay_paths],
            out_dir / "latent_overlay_original_vs_pseudo.png",
        )
        save_channel_diff_grid(
            latent_orig_full,
            latent_pseudo_full,
            orig_out.input,
            overlay_fn=create_latent_overlay,
            out_path=out_dir / "latent_channel_diff.png",
        )

        if args.adjust_latent:
            scale_values = build_scale_sequence(float(args.latent_scale),max_scale=args.latent_max_scale)
            panel_tensors = [orig_out.input, pseudo_out.input]
            panel_labels = ["Original", "Pseudo"]
            for scale in scale_values:
                scale_val = float(scale)
                adjust_map = (latent_pseudo - latent_orig).unsqueeze(0) * scale_val
                recon_adjusted = adjust_latent_and_decode(
                    multimap_model,
                    latent_spatial=orig_out.latent_spatial,
                    adjust_map=adjust_map,
                ).squeeze(0).clamp(0, 1)
                panel_tensors.append(recon_adjusted)
                panel_labels.append(f"Adjust x{scale_val:g}")
            if len(panel_tensors) > 2:
                save_adjustment_panel(panel_tensors, panel_labels, out_dir / "adjustment_panel.png")

    print("[pipeline] processing complete.")


if __name__ == "__main__":
    main()
