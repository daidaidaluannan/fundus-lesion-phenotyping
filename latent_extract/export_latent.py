#!/usr/bin/env python3
"""
批量提取“原图 vs pseudo-healthy” 的 multimap 空间 latent 差异，并写入 CSV。
- 每张图：AOT-GAN 生成 pseudo-healthy，二者喂入 multimap，计算归一化 latent 差值。
- 可选按通道做 mean/max 聚合后再展平；默认 mean，因此 7×7=49 个元素。
- 指定 --top-k 时，按差值幅度取前 k%。
"""
import argparse
import csv
import math
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
repo_root = Path(__file__).resolve().parent.parent
import sys
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
from model_zoo.phanes import AnomalyMap  # noqa: E402
from transforms.synthetic import GenerateMasks  # noqa: E402
import train_models.aotgan_1123 as aot  # noqa: E402
from counterfactual_pipeline import load_image  # noqa: E402
from train_models.visualize_multimap_single_1113 import (
    ModelConfig,
    build_multimap_model,
    normalize_tensor,
)


def list_images(image_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def compute_keep_count(total: int, top_k: float) -> int:
    if top_k is None:
        return total
    if top_k <= 0:
        raise ValueError("top-k must be > 0 when specified.")
    pct = max(0.0, min(float(top_k), 100.0))
    return max(1, int(math.ceil(total * (pct / 100.0))))


def build_aot_models(
    aae_ckpt: Path,
    aot_ckpt: Path,
    color_mode: str,
    device: torch.device,
):
    ret_aae = aot.load_ret_aae(aae_ckpt, device)
    netG, _ = aot.build_aotgan_networks(color_mode, device)
    ckpt = torch.load(aot_ckpt, map_location=device)
    state = ckpt.get("netG", ckpt)
    netG.load_state_dict(state)
    netG.eval()
    ret_aae.eval()

    ano = AnomalyMap()
    mask_generator = GenerateMasks(min_size=20, max_size=40)
    return ret_aae, netG, ano, mask_generator


@torch.no_grad()
def run_aot_once(
    ret_aae,
    netG,
    ano,
    mask_generator,
    image_tensor: torch.Tensor,
    mask_threshold: float,
    color_mode: str,
) -> torch.Tensor:
    """
    输入 (1,C,H,W) tensor，输出 pseudo-healthy (1,3,H,W)。
    """
    rgb = image_tensor
    coarse_rgb, _ = ret_aae(rgb)
    x_gray = TF.rgb_to_grayscale(rgb)
    coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
    mask = aot.generate_mask(
        ano,
        mask_generator,
        x_gray,
        coarse_gray,
        mask_threshold=mask_threshold,
        add_synthetic=False,
    )

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
    return pred


@torch.no_grad()
def latent_diff_vector(
    multimap_model: torch.nn.Module,
    orig_img: torch.Tensor,
    pseudo_img: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    返回归一化 latent 差的展平向量；reduction 在通道维做 mean/max。
    """
    out_orig = multimap_model(orig_img)
    out_pseudo = multimap_model(pseudo_img)

    if isinstance(out_orig, (list, tuple)):
        if len(out_orig) == 2:
            _, latent_orig = out_orig
        elif len(out_orig) == 3:
            _, _, latent_orig = out_orig
        else:
            raise RuntimeError(f"Unsupported multimap output length: {len(out_orig)}")
    else:
        raise RuntimeError("Multimap model must return recon & latent.")

    if isinstance(out_pseudo, (list, tuple)):
        if len(out_pseudo) == 2:
            _, latent_pseudo = out_pseudo
        elif len(out_pseudo) == 3:
            _, _, latent_pseudo = out_pseudo
        else:
            raise RuntimeError(f"Unsupported multimap output length: {len(out_pseudo)}")
    else:
        raise RuntimeError("Multimap model must return recon & latent.")

    if latent_orig.ndim != 4 or latent_pseudo.ndim != 4:
        raise RuntimeError("Expected spatial latents (B,C,H,W).")
    if latent_orig.shape != latent_pseudo.shape:
        latent_pseudo = F.interpolate(
            latent_pseudo,
            size=latent_orig.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    mm_norm = normalize_tensor(latent_orig)
    ps_norm = normalize_tensor(latent_pseudo)
    diff = torch.abs(mm_norm - ps_norm)
    if reduction == "max":
        diff_map = diff.max(dim=1, keepdim=True).values
    else:
        diff_map = diff.mean(dim=1, keepdim=True)
    diff_flat = diff_map.squeeze(0).reshape(-1)
    return diff_flat.detach().cpu()


def select_values(flat: torch.Tensor, keep_count: int) -> torch.Tensor:
    if keep_count >= flat.numel():
        return flat
    vals, _ = torch.topk(flat, keep_count, largest=True)
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch export multimap spatial latent vectors to CSV.")
    p.add_argument("--image-dir", required=True, help="目录，包含待处理图片。")
    p.add_argument("--out-csv", default=None, help="输出 CSV 路径，默认写到 image-dir/latents.csv。")
    p.add_argument("--multimap-ckpt", required=True, help="AAE_S / multimap 权重路径。")
    p.add_argument("--multimap-config", default=None, help="AAE_S 配置 JSON（可选）。")
    p.add_argument("--aot-img-size", type=int, default=224, help="AOT-GAN 入口尺寸。")
    p.add_argument("--viz-img-size", type=int, default=512, help="multimap 输入尺寸（会将原图与 pseudo 上采样到此尺寸）。")
    p.add_argument("--device", default="cuda", help="运行设备，默认 cuda，可设 cpu。")
    p.add_argument("--aae-c-ckpt", required=True, help="预训练 AAE_C (Ret_AAE) 权重路径。")
    p.add_argument("--aot-ckpt", required=True, help="AOT-GAN netG 权重路径。")
    p.add_argument("--mask-threshold", type=float, default=0.153, help="AOT-GAN 掩膜阈值。")
    p.add_argument("--color-mode", choices=["gray", "color"], default="gray", help="AOT-GAN 训练模式。")
    p.add_argument("--reduction", choices=["mean", "max"], default="mean", help="通道聚合方式，默认 mean 得到 7x7=49 个差值。")
    p.add_argument(
        "--top-k",
        type=float,
        default=None,
        help="保留幅值最大的前 k%% latent 元素（0-100]；不指定则导出全部元素。",
    )
    return p


def main(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image-dir not found: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("[export_latent_csv] CUDA unavailable, falling back to CPU.")

    multimap_model, cfg = build_multimap_model(
        ModelConfig(ckpt_path=args.multimap_ckpt, config_path=args.multimap_config),
        device=device,
    )
    multimap_model.eval()

    ret_aae, netG, ano, mask_generator = build_aot_models(
        aae_ckpt=Path(args.aae_c_ckpt),
        aot_ckpt=Path(args.aot_ckpt),
        color_mode=args.color_mode,
        device=device,
    )

    keep_count = None
    total_elements = None
    header = None
    rows: List[Sequence] = []

    for idx, img_path in enumerate(images):
        aot_in = load_image(Path(img_path), size=args.aot_img_size, device=device)
        pseudo = run_aot_once(
            ret_aae=ret_aae,
            netG=netG,
            ano=ano,
            mask_generator=mask_generator,
            image_tensor=aot_in,
            mask_threshold=args.mask_threshold,
            color_mode=args.color_mode,
        )

        orig_viz = F.interpolate(
            aot_in,
            size=(args.viz_img_size, args.viz_img_size),
            mode="bilinear",
            align_corners=False,
        )
        pseudo_viz = F.interpolate(
            pseudo,
            size=(args.viz_img_size, args.viz_img_size),
            mode="bilinear",
            align_corners=False,
        )

        diff_flat = latent_diff_vector(
            multimap_model=multimap_model,
            orig_img=orig_viz.to(device),
            pseudo_img=pseudo_viz.to(device),
            reduction=args.reduction,
        )

        if keep_count is None:
            total_elements = diff_flat.numel()
            keep_count = compute_keep_count(total_elements, args.top_k)
            header = ["image"] + [f"latent{i}" for i in range(1, keep_count + 1)]

        vec = select_values(diff_flat, keep_count)
        rows.append([str(img_path)] + vec.tolist())

    out_csv = Path(args.out_csv) if args.out_csv else image_dir / "latents.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header or [])
        writer.writerows(rows)

    print(f"[export_latent_csv] wrote {len(rows)} rows to {out_csv}")
    print(f"[export_latent_csv] latent elements per image: {keep_count} (total {total_elements})")


if __name__ == "__main__":
    main(build_parser().parse_args())
