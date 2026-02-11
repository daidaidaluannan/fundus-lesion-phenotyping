#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jupyter 辅助脚本：加载训练好的 RET-AAE 与 AOT-GAN 模型，对单张图像自动生成掩膜并输出修复结果。
可以在 notebook 中 import 并调用 `run_inference_demo(...)` 获取推理可视化。
"""
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from model_zoo.aotgan.aotgan import InpaintGenerator, Discriminator  # noqa: E402
from model_zoo.aotgan.loss import loss as loss_module  # noqa: E402
from model_zoo.phanes import AnomalyMap  # noqa: E402
from transforms.synthetic import GenerateMasks  # noqa: E402
from train_models.AAE_C import AAE  # noqa: E402
from train_models.aotgan_1123 import (  # noqa: E402
    _infer_upsample_mode,
    load_ret_aae,
    build_aotgan_networks,
    generate_mask,
)


def load_image(image_path: Path, img_size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)  # (1,3,H,W)


def load_inpaint_generator(checkpoint_path: Path,
                           color_mode: Optional[str],
                           device: torch.device):
    """
    加载 AOT-GAN 生成器。
    若 color_mode 为 None，则优先读取 checkpoint['stats']['color_mode']；
    如果两者冲突，会打印告警但仍按用户指定模式重建网络。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_color_mode = ckpt.get("stats", {}).get("color_mode")
    if color_mode is None:
        effective_mode = ckpt_color_mode or "gray"
    else:
        effective_mode = color_mode
        if ckpt_color_mode and ckpt_color_mode != color_mode:
            print(f"[Warn] Checkpoint color_mode={ckpt_color_mode}, "
                  f"but user specified {color_mode}. Using {color_mode}.")
    netG, _ = build_aotgan_networks(effective_mode, device)
    netG.load_state_dict(ckpt["netG"], strict=True)
    netG.eval()
    return netG, effective_mode


def prepare_demo_models(ret_aae_path: Path,
                        aot_ckpt_path: Path,
                        color_mode: Optional[str],
                        device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ret_aae = load_ret_aae(ret_aae_path, device)
    netG, effective_mode = load_inpaint_generator(aot_ckpt_path, color_mode, device)
    ano = AnomalyMap()
    return ret_aae, netG, ano, device, effective_mode


def run_inference_demo(image_path: str,
                       ret_aae_path: str,
                       aot_ckpt_path: str,
                       color_mode: Optional[str] = None,
                       mask_threshold: float = 0.153,
                       img_size: int = 224,
                       device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    ret_aae, netG, ano, device, effective_mode = prepare_demo_models(
        Path(ret_aae_path),
        Path(aot_ckpt_path),
        color_mode,
        device,
    )
    sample = load_image(Path(image_path), img_size=img_size).to(device)
    with torch.no_grad():
        coarse_rgb, _ = ret_aae(sample)
        x_gray = TF.rgb_to_grayscale(sample)
        coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
        mask = generate_mask(ano, None, x_gray, coarse_gray,
                             mask_threshold=mask_threshold,
                             add_synthetic=False)
        if effective_mode == "color":
            mask_rgb = mask.repeat(1, 3, 1, 1)
            transformed = (sample * (1 - mask_rgb)) + mask_rgb
            pred = netG(transformed, mask)
            pred_rgb = pred.clamp(0.0, 1.0)
            mask_vis = mask_rgb
        else:
            transformed = (x_gray * (1 - mask)) + mask
            pred = netG(transformed, mask)
            rgb_gray = TF.rgb_to_grayscale(sample)
            rgb_gray_exp = rgb_gray.expand_as(sample)
            ratio = torch.where(rgb_gray_exp > 1e-6,
                                sample / torch.clamp(rgb_gray_exp, min=1e-6),
                                torch.ones_like(sample))
            pred_rgb = torch.clamp(pred.repeat(1, 3, 1, 1) * ratio, 0.0, 1.0)
            mask_vis = mask.repeat(1, 3, 1, 1)
    return {
        "input": sample.cpu(),
        "coarse": coarse_rgb.cpu(),
        "mask": mask_vis.cpu(),
        "prediction": pred_rgb.cpu(),
    }


def plot_demo_results(results: Dict[str, torch.Tensor], figsize=(12, 6),fig_path = Path("aotgan_inference_demo_result.png")):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        try:
            plt.switch_backend("module://matplotlib_inline.backend_inline")
        except Exception:
            print("[Warn] 当前 Matplotlib backend 为 Agg，无法直接显示图像。"
                  "请在 Notebook 中执行 `%matplotlib inline` 或手动保存图像。")
    tensors = [results["input"], results["coarse"], results["prediction"], results["mask"]]
    grid = make_grid(torch.cat(tensors, dim=0), nrow=1, normalize=True, value_range=(0, 1))
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Input | RET-AAE coarse | AOT prediction | Mask")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()


__all__ = [
    "load_image",
    "load_inpaint_generator",
    "prepare_demo_models",
    "run_inference_demo",
    "plot_demo_results",
]
