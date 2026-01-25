#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual comparison tool for retinal autoencoders.

This script performs reconstruction & latent-space comparisons between a
Ret_AAE_multimap model and a baseline Ret_AAE (0901 or 1028 variant).
It generates difference overlays, latent overlays, and optional decoder-based
comparisons, now with FOV masking to suppress background artefacts.

Author: GPT-5 Codex
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.cm as cm

from AAE_S import AAE as MultiMapAAE
from AAE_C import AAE as BaselineAAE0901
#from Ret_AAE_1028 import AAE as BaselineAAE1028
#from train_latent_decoder_1101 import LatentDecoder  # reuse lightweight decoder
from train_transformer_decoder import TransformerLatentDecoder


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] per-tensor."""
    if t.numel() == 0:
        return t
    t_min = t.amin(dim=tuple(range(1, t.ndim)), keepdim=True)
    t_max = t.amax(dim=tuple(range(1, t.ndim)), keepdim=True)
    denom = (t_max - t_min).clamp(min=1e-6)
    return (t - t_min) / denom


def load_image(
    path: str,
    img_size: int,
    grayscale: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load an image file and convert to tensor (C, H, W) normalized to [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("L" if grayscale else "RGB")
        img = img.resize((img_size, img_size), Image.BICUBIC)

    transform_ops = [transforms.ToTensor()]
    if grayscale:
        transform_ops.insert(0, transforms.Grayscale(num_output_channels=1))
    transform = transforms.Compose(transform_ops)

    tensor = transform(img).to(device)
    return tensor


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    """Save tensor image (C,H,W) or (H,W) in [0,1] to disk."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.clamp(0, 1)
    grid = vutils.make_grid(tensor, nrow=1, normalize=False)
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(ndarr.squeeze() if ndarr.shape[2] == 1 else ndarr).save(path)


def save_latent_grid(tensor: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Save latent spatial tensor as a grid image."""
    tensor = tensor.detach().cpu()
    tensor = normalize_tensor(tensor)
    vutils.save_image(tensor, path, nrow=nrow, normalize=False)


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


def _load_state_dict(model: torch.nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load state dict with logging for missing/unexpected keys."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])

    if missing:
        print(f"[load_state_dict] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys ({len(unexpected)}): {unexpected}")
    print(f"[load_state_dict] Loaded weights from {ckpt_path}")


# ---------------------------------------------------------------------------
# FOV mask construction
# ---------------------------------------------------------------------------

def build_fov_mask(
    original: torch.Tensor,
    threshold: float = 0.05,
    erode_kernel: int = 7,
) -> torch.Tensor:
    """
    Build a field-of-view mask to suppress black background in overlays.

    Args:
        original: input image tensor (C, H, W) in [0, 1].
        threshold: gray-level threshold to distinguish foreground.
        erode_kernel: odd integer kernel size for morphological erosion.

    Returns:
        mask tensor (H, W) with values in [0, 1].
    """
    if original.ndim != 3:
        raise ValueError("build_fov_mask expects a 3D tensor (C, H, W)")

    gray = original.mean(dim=0)  # (H, W)
    mask = (gray > threshold).float()

    if erode_kernel > 1:
        if erode_kernel % 2 == 0:
            raise ValueError("erode_kernel must be an odd integer")
        pad = erode_kernel // 2
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        eroded = -F.max_pool2d(-mask_4d, kernel_size=erode_kernel, stride=1, padding=pad)
        mask = eroded.squeeze()
        mask = mask.clamp(0.0, 1.0)

    return mask


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    ckpt_path: str
    config_path: Optional[str] = None
    variant: Optional[str] = None


def build_multimap_model(
    cfg: ModelConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiate and load the Ret_AAE_multimap model."""
    if cfg.config_path:
        with open(cfg.config_path, "r") as f:
            raw_config = json.load(f)
        config = raw_config.get("AAE_S", raw_config)
    else:
        config = {}

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model = MultiMapAAE(**config).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[build_multimap_model] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[build_multimap_model] Unexpected keys ({len(unexpected)}): {unexpected}")
    print(f"[build_multimap_model] Loaded weights from {cfg.ckpt_path}")
    model.eval()
    return model, config


def build_baseline_model(
    cfg: ModelConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiate and load the baseline Ret_AAE model variant."""
    if cfg.config_path:
        with open(cfg.config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    variant = (cfg.variant or "").lower()
    if variant == "0901":
        model = BaselineAAE0901(**config).to(device)
    #elif variant == "1028":
    #    model = BaselineAAE1028(**config).to(device)
    else:
        raise ValueError("Baseline variant must be '0901' or '1028'.")

    _load_state_dict(model, cfg.ckpt_path, strict=False)
    model.eval()
    return model, config


def build_transformer_decoder(
    ckpt_path: str,
    latent_dim: int,
    spatial_hw: Tuple[int, int],
    device: torch.device,
) -> TransformerLatentDecoder:
    """Load transformer latent decoder."""
    decoder = TransformerLatentDecoder(
        latent_dim=latent_dim,
        spatial_hw=spatial_hw,
    ).to(device)

    _load_state_dict(decoder, ckpt_path, strict=False)
    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def create_diff_overlay(
    original: torch.Tensor,
    multimap_diff: torch.Tensor,
    baseline_diff: torch.Tensor,
    out_path: str,
    alpha: float = 0.5,
    threshold: float = 0.0,
    gamma: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Create overlay heatmap of difference delta, optionally masked by FOV."""
    original = original.detach().cpu()
    multimap_diff = multimap_diff.detach().cpu()
    baseline_diff = baseline_diff.detach().cpu()

    if original.ndim != 3:
        raise ValueError("original tensor must be 3D (C, H, W)")
    if multimap_diff.shape != baseline_diff.shape:
        raise ValueError("multimap_diff and baseline_diff must share the same shape")

    if original.shape[0] == 1:
        original = original.repeat(3, 1, 1)

    diff_delta = torch.abs(multimap_diff - baseline_diff)

    if threshold > 0.0:
        diff_delta = torch.clamp(diff_delta - threshold, min=0.0)
    if gamma != 1.0:
        diff_delta = diff_delta.pow(gamma)

    if mask is not None:
        mask = mask.to(diff_delta.device)
        diff_delta = diff_delta * mask
        valid = mask > 0
        if valid.any():
            vals = diff_delta[valid]
            vals_min = vals.min()
            vals_max = vals.max()
            diff_delta = torch.zeros_like(diff_delta)
            diff_delta[valid] = (vals - vals_min) / (vals_max - vals_min + 1e-6)
        else:
            diff_delta = torch.zeros_like(diff_delta)
    else:
        diff_delta = normalize_tensor(diff_delta.unsqueeze(0)).squeeze(0)

    if diff_delta.shape[1:] != original.shape[1:]:
        diff_delta = F.interpolate(
            diff_delta.unsqueeze(0),
            size=original.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    heat_map = diff_delta.squeeze(0).numpy()
    orig_np = original.permute(1, 2, 0).numpy()
    orig_uint8 = (orig_np * 255.0).clip(0, 255).astype(np.uint8)
    colored = cm.get_cmap("jet")(heat_map)[..., :3]
    colored_uint8 = (colored * 255.0).astype(np.uint8)

    overlay = (1.0 - alpha) * orig_uint8.astype(np.float32) + alpha * colored_uint8.astype(np.float32)
    overlay_uint8 = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8).save(out_path)
    print(f"[create_diff_overlay] saved overlay to {out_path}")


def create_latent_overlay(
    original: torch.Tensor,
    multimap_latent: torch.Tensor,
    baseline_latent: torch.Tensor,
    out_path: str,
    alpha: float = 0.5,
    threshold: float = 0.0,
    gamma: float = 1.0,
    reduction: str = "mean",
    fusion_mode: Optional[str] = None,
    fusion_weight: float = 0.5,
    top_pct: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Create overlay between two latent spatial tensors, with masking."""
    original = original.detach().cpu()
    multimap_latent = multimap_latent.detach().cpu()
    baseline_latent = baseline_latent.detach().cpu()

    if original.ndim != 3:
        raise ValueError("original tensor must be 3D (C, H, W)")
    if multimap_latent.shape != baseline_latent.shape:
        raise ValueError("Latent maps must share identical shapes for comparison.")

    if original.shape[0] == 1:
        original = original.repeat(3, 1, 1)

    mm_norm = normalize_tensor(multimap_latent.unsqueeze(0)).squeeze(0)
    bl_norm = normalize_tensor(baseline_latent.unsqueeze(0)).squeeze(0)
    diff = torch.abs(mm_norm - bl_norm)
    channel_scores = diff.reshape(diff.size(0), -1).mean(dim=1)

    selected_diff = diff
    if top_pct is not None:
        pct_raw = float(top_pct)
        if pct_raw <= 0:
            raise ValueError("top_pct must be greater than 0 when provided.")
        pct_clamped = min(pct_raw, 100.0)
        keep_ratio = pct_clamped / 100.0
        keep_count = max(1, int(math.ceil(diff.size(0) * keep_ratio)))
        keep_count = min(keep_count, diff.size(0))
        top_vals, top_idx = torch.topk(channel_scores, keep_count, largest=True)
        selected_diff = diff[top_idx]
        display_cap = min(8, keep_count)
        idx_list = top_idx.cpu().tolist()
        score_list = [round(float(v), 6) for v in top_vals.cpu().tolist()]
        print(
            f"[latent overlay] top_pct {pct_raw}: keeping {keep_count}/{diff.size(0)} channels "
            f"(≈ {keep_ratio * 100:.2f}% of total)"
        )
        print(f"    top channel indices: {idx_list[:display_cap]}")
        print(f"    top channel scores:  {score_list[:display_cap]}")
        if keep_count > display_cap:
            print(f"    ... ({keep_count - display_cap} more channels truncated in log)")
    else:
        print("[latent overlay] top_pct not set, using all channels")

    mean_map = selected_diff.mean(dim=0, keepdim=True).unsqueeze(0)
    max_map = selected_diff.max(dim=0, keepdim=True).values.unsqueeze(0)
    fuse_w = float(abs(fusion_weight))

    if fusion_mode == "linear":
        diff_map = (1 - fuse_w) * mean_map + fuse_w * max_map
    elif fusion_mode == "gated":
        gate = torch.sigmoid(fuse_w * (max_map - mean_map))
        diff_map = (1 - gate) * mean_map + gate * max_map
    else:  # default branch
        diff_map = max_map if reduction == "max" else mean_map

    if threshold > 0.0:
        diff_map = torch.clamp(diff_map - threshold, min=0.0)
    if gamma != 1.0:
        diff_map = diff_map.pow(gamma)

    diff_map = diff_map.squeeze(0)

    if mask is not None:
        mask = mask.to(diff_map.device)
        diff_map = diff_map * mask
        valid = mask > 0
        if valid.any():
            vals = diff_map[valid]
            vals_min = vals.min()
            vals_max = vals.max()
            diff_map = torch.zeros_like(diff_map)
            diff_map[valid] = (vals - vals_min) / (vals_max - vals_min + 1e-6)
        else:
            diff_map = torch.zeros_like(diff_map)
    else:
        diff_map = normalize_tensor(diff_map.unsqueeze(0)).squeeze(0)

    if diff_map.shape[1:] != original.shape[1:]:
        diff_map = F.interpolate(
            diff_map.unsqueeze(0),
            size=original.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    heat_map = diff_map.squeeze(0).numpy()
    orig_np = original.permute(1, 2, 0).numpy()
    orig_uint8 = (orig_np * 255.0).clip(0, 255).astype(np.uint8)
    colored = cm.get_cmap("jet")(heat_map)[..., :3]
    colored_uint8 = (colored * 255.0).astype(np.uint8)

    overlay = (1.0 - alpha) * orig_uint8.astype(np.float32) + alpha * colored_uint8.astype(np.float32)
    overlay_uint8 = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8).save(out_path)
    print(f"[create_latent_overlay] saved overlay to {out_path}")


def stack_images_vertically(
    image_paths: List[str],
    out_path: str,
) -> None:
    """Stack images vertically and save the combined image."""
    images = [Image.open(p) for p in image_paths if os.path.exists(p)]
    if not images:
        raise ValueError("No valid images provided for stacking.")

    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]

    combined.save(out_path)
    print(f"[stack_images_vertically] saved stacked image to {out_path}")


def combine_overlays(
    overlays: List[str],
    out_path: str,
    grid_size: Tuple[int, int] = (2, 2),
) -> None:
    """Combine a list of overlays into a grid (default 2x2)."""
    rows, cols = grid_size
    if len(overlays) != rows * cols:
        raise ValueError(f"Expected {rows * cols} overlays, got {len(overlays)}")

    imgs = [Image.open(p) for p in overlays]
    widths, heights = zip(*(img.size for img in imgs))
    max_width = max(widths)
    max_height = max(heights)

    combined = Image.new("RGB", (cols * max_width, rows * max_height))
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        combined.paste(img, (c * max_width, r * max_height))

    combined.save(out_path)
    print(f"[combine_overlays] saved combined grid to {out_path}")


# ---------------------------------------------------------------------------
# Model-specific visualization pipelines
# ---------------------------------------------------------------------------

@dataclass
class VisualizationOutputs:
    input: torch.Tensor
    recon: torch.Tensor
    diff: torch.Tensor
    latent_spatial: Optional[torch.Tensor] = None
    latent_global: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = None


def visualize_multimap(
    model: torch.nn.Module,
    image: torch.Tensor,
    out_dir: str,
) -> VisualizationOutputs:
    """Run inference on multimap model and save outputs."""
    ensure_dir(out_dir)

    with torch.no_grad():
        recon, latent = model(image.unsqueeze(0))

    recon = recon.squeeze(0).clamp(0, 1)
    diff = torch.abs(image - recon)
    latent_mean = latent.mean(dim=1, keepdim=True) if latent.ndim == 4 else latent
    latent_max = latent.max(dim=1, keepdim=True).values if latent.ndim == 4 else None

    save_tensor_image(image, os.path.join(out_dir, "input.png"))
    save_tensor_image(recon, os.path.join(out_dir, "recon.png"))
    save_tensor_image(normalize_tensor(diff.unsqueeze(0)).squeeze(0), os.path.join(out_dir, "diff.png"))

    if latent.ndim == 4:
        save_latent_grid(latent_mean, os.path.join(out_dir, "latent_mean.png"))
        save_latent_grid(latent_max, os.path.join(out_dir, "latent_max.png"))
        save_latent_grid(latent, os.path.join(out_dir, "latent_full.png"))

    return VisualizationOutputs(
        input=image,
        recon=recon,
        diff=diff,
        latent_spatial=latent if latent.ndim == 4 else None,
        latent_global=None,
        extra={"latent_mean": latent_mean, "latent_max": latent_max},
    )


def visualize_baseline(
    model: torch.nn.Module,
    image: torch.Tensor,
    out_dir: str,
    spatial_hw: Optional[Tuple[int, int]] = None,
) -> VisualizationOutputs:
    """Run inference on baseline model and save outputs."""
    ensure_dir(out_dir)

    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
    if isinstance(outputs, tuple):
        recon, latent = outputs
    else:
        recon = outputs
        latent = None

    recon = recon.squeeze(0).clamp(0, 1)
    diff = torch.abs(image - recon)

    save_tensor_image(image, os.path.join(out_dir, "input.png"))
    save_tensor_image(recon, os.path.join(out_dir, "recon.png"))
    save_tensor_image(normalize_tensor(diff.unsqueeze(0)).squeeze(0), os.path.join(out_dir, "diff.png"))

    latent_spatial = None
    latent_global = None

    if latent is not None:
        if latent.ndim == 2:
            latent_global = latent
            torch.save(latent, os.path.join(out_dir, "latent_global.pt"))
        elif latent.ndim == 4:
            latent_spatial = latent
            latent_mean = latent.mean(dim=1, keepdim=True)
            latent_max = latent.max(dim=1, keepdim=True).values
            save_latent_grid(latent_mean, os.path.join(out_dir, "latent_mean.png"))
            save_latent_grid(latent_max, os.path.join(out_dir, "latent_max.png"))
            torch.save(latent, os.path.join(out_dir, "latent_spatial.pt"))
        else:
            torch.save(latent, os.path.join(out_dir, "latent.pt"))

    # Additional optional handling for spatial_hw, if provided but latent missing
    if latent_spatial is None and spatial_hw is not None and latent_global is not None:
        print("[visualize_baseline] spatial latent missing, available only as global latent.")

    return VisualizationOutputs(
        input=image,
        recon=recon,
        diff=diff,
        latent_spatial=latent_spatial,
        latent_global=latent_global,
        extra={},
    )


# ---------------------------------------------------------------------------
# Latent decoder integration
# ---------------------------------------------------------------------------

def decode_latent_with_transformer(
    decoder: TransformerLatentDecoder,
    latent_global: torch.Tensor,
) -> torch.Tensor:
    """Decode global latent to spatial latent using transformer decoder."""
    with torch.no_grad():
        decoded = decoder(latent_global)
    return decoded


def adjust_latent_and_decode(
    multimap_model: torch.nn.Module,
    latent_spatial: torch.Tensor,
    adjust_map: torch.Tensor,
) -> torch.Tensor:
    """
    Adjust multimap latent by adding adjust_map, then decode via model.decode.
    """
    if not hasattr(multimap_model, "decode"):
        raise AttributeError("multimap model lacks decode() function required for latent adjustment.")

    latent_adjusted = latent_spatial + adjust_map
    with torch.no_grad():
        recon = multimap_model.decode(latent_adjusted)
    return recon


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retinal AAE comparison tool with FOV masking.")

    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--img-size", type=int, default=512, help="Input resize (square).")
    parser.add_argument("--grayscale", action="store_true", help="Treat input as grayscale image.")

    parser.add_argument("--multimap-ckpt", required=True, help="Checkpoint for Ret_AAE_multimap.")
    parser.add_argument("--multimap-config", default=None, help="JSON config for multimap model.")

    parser.add_argument("--baseline-variant", choices=["0901", "1028"], required=True, help="Baseline variant.")
    parser.add_argument("--baseline-ckpt", required=True, help="Checkpoint for baseline model.")
    parser.add_argument("--baseline-config", default=None, help="JSON config for baseline model.")

    parser.add_argument("--decoder-ckpt", default=None, help="Optional transformer latent decoder checkpoint.")

    parser.add_argument("--out-dir", required=True, help="Output directory for visualizations.")

    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--heatmap-alpha", type=float, default=0.5, help="Overlay alpha for heatmaps.")
    parser.add_argument("--diff-threshold", type=float, default=0.0, help="Threshold to suppress small differences.")
    parser.add_argument("--diff-gamma", type=float, default=1.0, help="Gamma correction for difference maps.")

    parser.add_argument("--latent-top-pct", type=float, default=None, help="Top percentile of channels to keep.")
    parser.add_argument("--latent-reduction", choices=["mean", "max"], default="mean", help="Reduction mode.")
    parser.add_argument("--latent-fusion", choices=[None, "linear", "gated"], default=None, help="Fusion mode.")
    parser.add_argument("--latent-fusion-weight", type=float, default=0.5, help="Fusion weight for linear/gated modes.")

    parser.add_argument("--adjust-latent", action="store_true", help="Adjust latent via difference map and decode.")
    parser.add_argument("--latent-scale", type=float, default=1.0, help="Scalar to multiply latent adjustments.")

    parser.add_argument("--mask-threshold", type=float, default=0.05,
                        help="FOV mask threshold (0-1) to suppress background.")
    parser.add_argument("--mask-erode", type=int, default=7,
                        help="FOV mask erosion kernel size (odd integer).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("[main] CUDA unavailable, falling back to CPU.")

    ensure_dir(args.output_dir)

    # Load input image
    img_tensor = load_image(
        args.image,
        img_size=args.img_size,
        grayscale=args.grayscale,
        device=device,
    )

    # Build models
    multimap_model, multimap_cfg = build_multimap_model(
        ModelConfig(
            ckpt_path=args.multimap_ckpt,
            config_path=args.multimap_config,
        ),
        device=device,
    )

    baseline_model, baseline_cfg = build_baseline_model(
        ModelConfig(
            ckpt_path=args.baseline_ckpt,
            config_path=args.baseline_config,
            variant=args.baseline_variant,
        ),
        device=device,
    )

    decoder = None
    decoder_spatial = None
    if args.decoder_ckpt:
        # infer latent_dim and spatial_hw from configs
        latent_dim = baseline_cfg.get("latent_dim") or baseline_cfg.get("z_dim")
        if latent_dim is None:
            raise ValueError("Baseline config must include latent_dim or z_dim for decoder.")
        # deduce spatial size from multimap config
        img_size = multimap_cfg.get("img_size", args.img_size)
        num_down = multimap_cfg.get("num_blocks", 5)
        latent_h = multimap_cfg.get("latent_h") or img_size // (2 ** num_down)
        latent_w = multimap_cfg.get("latent_w") or img_size // (2 ** num_down)
        decoder_spatial = (latent_h, latent_w)

        decoder = build_transformer_decoder(
            ckpt_path=args.decoder_ckpt,
            latent_dim=int(latent_dim),
            spatial_hw=decoder_spatial,
            device=device,
        )

    # Visualize models
    multimap_out = visualize_multimap(
        multimap_model,
        img_tensor,
        out_dir=os.path.join(args.output_dir, "multimap"),
    )
    baseline_out = visualize_baseline(
        baseline_model,
        img_tensor,
        out_dir=os.path.join(args.output_dir, "baseline"),
        spatial_hw=decoder_spatial,
    )

    # Create FOV mask
    fov_mask = build_fov_mask(
        multimap_out.input,
        threshold=args.mask_threshold,
        erode_kernel=args.mask_erode,
    )
    torch.save(fov_mask, os.path.join(args.output_dir, "fov_mask.pt"))
    save_tensor_image(fov_mask.unsqueeze(0), os.path.join(args.output_dir, "fov_mask.png"))

    # Prepare diffs for overlay
    multimap_diff = multimap_out.diff.mean(dim=0, keepdim=True)
    baseline_diff = baseline_out.diff.mean(dim=0, keepdim=True)

    if multimap_diff.shape != baseline_diff.shape:
        baseline_diff = F.interpolate(
            baseline_diff.unsqueeze(0),
            size=multimap_diff.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    create_diff_overlay(
        original=multimap_out.input,
        multimap_diff=multimap_diff,
        baseline_diff=baseline_diff,
        out_path=os.path.join(args.output_dir, "diff_overlay.png"),
        alpha=args.heatmap_alpha,
        threshold=args.diff_threshold,
        gamma=args.diff_gamma,
        mask=fov_mask,
    )

    # Latent overlays
    overlay_paths: List[str] = []

    if multimap_out.latent_spatial is not None and baseline_out.latent_spatial is not None:
        mm_latent = multimap_out.latent_spatial.mean(dim=1)
        bl_latent = baseline_out.latent_spatial.mean(dim=1)

        if mm_latent.shape != bl_latent.shape:
            bl_latent = F.interpolate(
                bl_latent.unsqueeze(0),
                size=mm_latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        overlay_path = os.path.join(args.output_dir, "latent_overlay_multimap_vs_baseline.png")
        create_latent_overlay(
            original=multimap_out.input,
            multimap_latent=mm_latent,
            baseline_latent=bl_latent,
            out_path=overlay_path,
            alpha=args.heatmap_alpha,
            threshold=args.diff_threshold,
            gamma=args.diff_gamma,
            reduction=args.latent_reduction,
            fusion_mode=args.latent_fusion,
            fusion_weight=args.latent_fusion_weight,
            top_pct=args.latent_top_pct,
            mask=fov_mask,
        )
        overlay_paths.append(overlay_path)

    # Decoder latent overlay
    decoded_latent = None
    if decoder is not None and baseline_out.latent_global is not None:
        decoded_latent = decode_latent_with_transformer(
            decoder,
            baseline_out.latent_global,
        )
        decoded_latent = decoded_latent.squeeze(0)

        if multimap_out.latent_spatial is not None:
            mm_latent = multimap_out.latent_spatial.mean(dim=1)
            overlay_path = os.path.join(args.output_dir, "latent_overlay_multimap_vs_decoder.png")
            create_latent_overlay(
                original=multimap_out.input,
                multimap_latent=mm_latent,
                baseline_latent=decoded_latent,
                out_path=overlay_path,
                alpha=args.heatmap_alpha,
                threshold=args.diff_threshold,
                gamma=args.diff_gamma,
                reduction=args.latent_reduction,
                fusion_mode=args.latent_fusion,
                fusion_weight=args.latent_fusion_weight,
                top_pct=args.latent_top_pct,
                mask=fov_mask,
            )
            overlay_paths.append(overlay_path)

    # Optional latent adjustment
    if args.adjust_latent and decoded_latent is not None and multimap_out.latent_spatial is not None:
        mm_latent = multimap_out.latent_spatial
        decoded_up = decoded_latent.unsqueeze(0)

        if decoded_up.shape != mm_latent.shape:
            decoded_up = F.interpolate(
                decoded_up,
                size=mm_latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        adjust_map = (decoded_up - mm_latent) * args.latent_scale
        recon_adjusted = adjust_latent_and_decode(
            multimap_model,
            latent_spatial=mm_latent,
            adjust_map=adjust_map,
        ).squeeze(0)

        save_tensor_image(
            recon_adjusted.clamp(0, 1),
            os.path.join(args.output_dir, "adjusted_recon.png"),
        )
        save_tensor_image(
            torch.abs(recon_adjusted - img_tensor).mean(dim=0, keepdim=True),
            os.path.join(args.output_dir, "adjusted_diff.png"),
        )

    # Stack overlays if available
    diff_overlay_path = os.path.join(args.output_dir, "diff_overlay.png")
    if diff_overlay_path in overlay_paths:
        overlay_paths.remove(diff_overlay_path)
    if os.path.exists(diff_overlay_path):
        overlay_paths.insert(0, diff_overlay_path)

    if len(overlay_paths) >= 2:
        try:
            combine_overlays(
                overlays=overlay_paths[:4] if len(overlay_paths) >= 4 else overlay_paths,
                out_path=os.path.join(args.output_dir, "overlay_grid.png"),
                grid_size=(2, 2) if len(overlay_paths) >= 4 else (len(overlay_paths), 1),
            )
        except Exception as exc:
            print(f"[main] combine_overlays skipped: {exc}")

    print("[main] processing complete.")


if __name__ == "__main__":
    main()
