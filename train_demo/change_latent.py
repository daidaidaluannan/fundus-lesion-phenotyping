#!/usr/bin/env python3
"""
潜变量迁移模块 (Latent Transfer)

功能：将异常图像的显著空间潜变量位置迁移到正常图像，生成带有异常特征的重建。

工作流程：
1. 异常图像 → AAE_C + AOT-GAN → 伪健康重建
2. 异常原图 & 伪健康 & 正常图 → AAE_S → 空间潜变量
3. 计算异常原图 vs 伪健康的 latent 差异图
4. 选取 Top-K% 显著位置，将异常潜变量复制到正常图对应位置
5. 解码生成带异常潜变量的正常图重建
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# ============================================================================
# Path Setup
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from model_zoo.phanes import AnomalyMap  # noqa: E402
from transforms.synthetic import GenerateMasks  # noqa: E402
from train_demo import aotgan_1123 as aot  # noqa: E402
from train_demo.visualize_multimap_single_1113 import (  # noqa: E402
    ModelConfig,
    build_multimap_model,
    ensure_dir,
    save_tensor_image,
)

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
)
logger = logging.getLogger("change_latent")

# ============================================================================
# Constants
# ============================================================================
DEFAULT_MASK_THRESHOLD = 0.153
DEFAULT_AOT_IMG_SIZE = 224
DEFAULT_MULTIMAP_IMG_SIZE = 512
DEFAULT_TOP_PERCENT = 10.0
DEFAULT_SEED = 42


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class AotGanConfig:
    """AOT-GAN 相关配置"""
    aae_ckpt: Path
    aot_ckpt: Path
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    color_mode: str = "gray"
    img_size: int = DEFAULT_AOT_IMG_SIZE


@dataclass
class MultimapConfig:
    """Multimap (AAE_S) 相关配置"""
    ckpt_path: Path
    config_path: Optional[Path] = None
    img_size: int = DEFAULT_MULTIMAP_IMG_SIZE


@dataclass
class LatentTransferConfig:
    """潜变量迁移配置"""
    top_percent: float = DEFAULT_TOP_PERCENT
    reduction: str = "mean"  # "mean" or "max"


@dataclass
class AotGanOutput:
    """AOT-GAN 推理输出"""
    input_rgb: torch.Tensor
    coarse: torch.Tensor
    mask: torch.Tensor
    mask_rgb: torch.Tensor
    pseudo: torch.Tensor


@dataclass
class MultimapOutput:
    """Multimap 推理输出"""
    recon: torch.Tensor
    diff: torch.Tensor
    latent: Optional[torch.Tensor]


@dataclass
class OutputPaths:
    """输出文件路径管理"""
    base_dir: Path

    # AOT-GAN 输出
    a_mask: Path = field(init=False)
    a_pseudo: Path = field(init=False)
    a_coarse: Path = field(init=False)

    # Multimap 输出
    a_input: Path = field(init=False)
    a_recon: Path = field(init=False)
    a_diff: Path = field(init=False)
    b_input: Path = field(init=False)
    b_recon: Path = field(init=False)
    b_recon_modified: Path = field(init=False)

    # 元数据
    top_positions: Path = field(init=False)

    def __post_init__(self):
        self.a_mask = self.base_dir / "a_mask.png"
        self.a_pseudo = self.base_dir / "a_pseudo.png"
        self.a_coarse = self.base_dir / "a_coarse.png"
        self.a_input = self.base_dir / "a_input.png"
        self.a_recon = self.base_dir / "a_recon.png"
        self.a_diff = self.base_dir / "a_diff.png"
        self.b_input = self.base_dir / "b_input.png"
        self.b_recon = self.base_dir / "b_recon.png"
        self.b_recon_modified = self.base_dir / "b_recon_modified.png"
        self.top_positions = self.base_dir / "top_positions.txt"


# ============================================================================
# Image Loading
# ============================================================================
def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """加载并预处理图像为张量。

    Args:
        path: 图像文件路径
        size: 目标尺寸 (正方形)
        device: 计算设备

    Returns:
        形状为 (1, 3, H, W) 的张量
    """
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)


# ============================================================================
# AOT-GAN Inference
# ============================================================================
@torch.no_grad()
def run_aotgan_inference(
    image_tensor: torch.Tensor,
    config: AotGanConfig,
    device: torch.device,
) -> AotGanOutput:
    """使用 AAE_C + AOT-GAN 生成自动掩膜与伪健康重建。

    Args:
        image_tensor: 输入图像张量 (1, 3, H, W)
        config: AOT-GAN 配置
        device: 计算设备

    Returns:
        包含 input、coarse、mask、pseudo 等的输出对象
    """
    # 加载模型
    ret_aae = aot.load_ret_aae(config.aae_ckpt, device)
    netG, _ = aot.build_aotgan_networks(config.color_mode, device)

    ckpt = torch.load(config.aot_ckpt, map_location=device)
    state = ckpt.get("netG", ckpt)
    netG.load_state_dict(state)
    netG.eval()

    # 初始化辅助模块
    ano = AnomalyMap()
    mask_generator = GenerateMasks(min_size=20, max_size=40)

    # 前向推理
    rgb = image_tensor.to(device)
    coarse_rgb, _ = ret_aae(rgb)

    x_gray = TF.rgb_to_grayscale(rgb)
    coarse_gray = TF.rgb_to_grayscale(coarse_rgb)

    mask = aot.generate_mask(
        ano, mask_generator, x_gray, coarse_gray,
        mask_threshold=config.mask_threshold,
        add_synthetic=False,
    )

    # 根据颜色模式处理
    if config.color_mode == "color":
        mask_rgb = mask.repeat(1, 3, 1, 1)
        transformed = rgb * (1 - mask_rgb) + mask_rgb
    else:
        mask_rgb = mask
        transformed = x_gray * (1 - mask) + mask

    pred = netG(transformed, mask)
    if config.color_mode == "gray":
        pred = pred.repeat(1, 3, 1, 1)
    pred = pred.clamp(0.0, 1.0)

    return AotGanOutput(
        input_rgb=rgb,
        coarse=coarse_rgb.clamp(0, 1),
        mask=mask,
        mask_rgb=mask_rgb,
        pseudo=pred,
    )


# ============================================================================
# Multimap Inference
# ============================================================================
@torch.no_grad()
def run_multimap(model: torch.nn.Module, image: torch.Tensor) -> MultimapOutput:
    """运行 Multimap (AAE_S) 模型。

    Args:
        model: Multimap 模型
        image: 输入图像张量

    Returns:
        包含 recon、diff、latent 的输出对象
    """
    recon, latent = model(image)
    recon = recon.clamp(0, 1)
    diff = torch.abs(image - recon)
    latent_spatial = latent if latent.ndim == 4 else None

    return MultimapOutput(recon=recon, diff=diff, latent=latent_spatial)


# ============================================================================
# Latent Space Operations
# ============================================================================
def compute_spatial_diff_map(
    latent_orig: torch.Tensor,
    latent_pseudo: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """计算跨通道聚合后的空间差异图。

    Args:
        latent_orig: 原图潜变量 (B, C, H, W)
        latent_pseudo: 伪健康潜变量 (B, C, H, W)
        reduction: 聚合方式 ("mean" 或 "max")

    Returns:
        空间差异图 (B, H, W)
    """
    if latent_orig.shape != latent_pseudo.shape:
        raise ValueError("latent_orig and latent_pseudo must share shape.")

    diff = (latent_orig - latent_pseudo).abs()

    if reduction == "max":
        return diff.max(dim=1).values
    return diff.mean(dim=1)


def pick_top_spatial_positions(
    diff_map: torch.Tensor,
    top_percent: float,
) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
    """在空间差异图上按百分比选择最显著的位置。

    Args:
        diff_map: 空间差异图 (B, H, W)
        top_percent: 选择的百分比 (0, 100]

    Returns:
        - spatial_mask: 空间掩膜 (B, 1, H, W)
        - positions: 每个 batch 的 (row, col) 位置列表
    """
    if not (0 < top_percent <= 100):
        raise ValueError("top_percent must be in (0, 100].")
    if diff_map.ndim != 3:
        raise ValueError("diff_map must be (B, H, W).")

    b, h, w = diff_map.shape
    total_positions = h * w
    k = max(1, min(math.ceil(total_positions * top_percent / 100.0), total_positions))

    # 找到 Top-K 位置
    flat = diff_map.view(b, -1)
    _, indices = torch.topk(flat, k, dim=1, largest=True)

    # 构建空间掩膜
    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_flat.scatter_(1, indices, True)
    mask = mask_flat.view(b, 1, h, w)

    # 提取位置坐标
    positions: List[List[Tuple[int, int]]] = []
    for batch_idx in range(b):
        batch_positions = [
            (idx // w, idx % w)
            for idx in indices[batch_idx].tolist()
        ]
        positions.append(batch_positions)

    return mask, positions


def replace_latent_positions(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    spatial_mask: torch.Tensor,
) -> torch.Tensor:
    """用空间掩膜控制的方式，将 source 对应位置的潜变量复制到 target。

    Args:
        source_latent: 源潜变量 (异常图)
        target_latent: 目标潜变量 (正常图)
        spatial_mask: 空间掩膜 (B, 1, H, W)

    Returns:
        更新后的潜变量
    """
    if source_latent.shape != target_latent.shape:
        raise ValueError("source_latent and target_latent must share shape.")
    if spatial_mask.ndim != 4:
        raise ValueError("spatial_mask must be (B, 1, H, W).")
    if spatial_mask.shape[0] != source_latent.shape[0]:
        raise ValueError("Batch dimension mismatch.")
    if spatial_mask.shape[-2:] != source_latent.shape[-2:]:
        raise ValueError("Spatial dimension mismatch.")

    mask_expanded = spatial_mask.to(dtype=torch.bool).expand_as(source_latent)
    return torch.where(mask_expanded, source_latent, target_latent)


# ============================================================================
# Pipeline Steps
# ============================================================================
def step1_aotgan_inference(
    abnormal_path: Path,
    config: AotGanConfig,
    device: torch.device,
    output_paths: OutputPaths,
) -> AotGanOutput:
    """步骤1: 对异常图像运行 AOT-GAN 推理。"""
    logger.info("Step 1: Running AOT-GAN inference...")

    img_tensor = load_image(abnormal_path, config.img_size, device)
    aot_out = run_aotgan_inference(img_tensor, config, device)

    # 保存中间结果
    save_tensor_image(aot_out.mask.squeeze(0), str(output_paths.a_mask))
    save_tensor_image(aot_out.pseudo.squeeze(0), str(output_paths.a_pseudo))
    save_tensor_image(aot_out.coarse.squeeze(0), str(output_paths.a_coarse))

    logger.info(f"  Saved: {output_paths.a_mask.name}, {output_paths.a_pseudo.name}")
    return aot_out


def step2_multimap_inference(
    aot_out: AotGanOutput,
    normal_path: Path,
    multimap_config: MultimapConfig,
    device: torch.device,
    output_paths: OutputPaths,
) -> Tuple[MultimapOutput, MultimapOutput, MultimapOutput]:
    """步骤2: 运行 Multimap 模型获取空间潜变量。"""
    logger.info("Step 2: Running Multimap inference...")

    # 构建模型
    model, _ = build_multimap_model(
        ModelConfig(
            ckpt_path=str(multimap_config.ckpt_path),
            config_path=str(multimap_config.config_path) if multimap_config.config_path else None,
        ),
        device=device,
    )
    model.eval()

    # 调整图像尺寸
    target_size = (multimap_config.img_size, multimap_config.img_size)
    img_a = F.interpolate(aot_out.input_rgb, size=target_size, mode="bilinear", align_corners=False)
    img_a_pseudo = F.interpolate(aot_out.pseudo, size=target_size, mode="bilinear", align_corners=False)
    img_b = load_image(normal_path, multimap_config.img_size, device)

    # 前向推理
    out_a = run_multimap(model, img_a)
    out_a_pseudo = run_multimap(model, img_a_pseudo)
    out_b = run_multimap(model, img_b)

    # 保存中间结果
    save_tensor_image(img_a.squeeze(0), str(output_paths.a_input))
    save_tensor_image(out_a.recon.squeeze(0), str(output_paths.a_recon))
    save_tensor_image(out_a.diff.squeeze(0), str(output_paths.a_diff))
    save_tensor_image(img_b.squeeze(0), str(output_paths.b_input))
    save_tensor_image(out_b.recon.squeeze(0), str(output_paths.b_recon))

    logger.info(f"  Saved: {output_paths.a_input.name}, {output_paths.b_input.name}")
    return out_a, out_a_pseudo, out_b, model


def step3_latent_transfer(
    out_a: MultimapOutput,
    out_a_pseudo: MultimapOutput,
    out_b: MultimapOutput,
    model: torch.nn.Module,
    transfer_config: LatentTransferConfig,
    output_paths: OutputPaths,
) -> None:
    """步骤3: 执行潜变量迁移并解码重建。"""
    logger.info("Step 3: Performing latent transfer...")

    latent_a = out_a.latent
    latent_a_pseudo = out_a_pseudo.latent
    latent_b = out_b.latent

    if latent_a is None or latent_a_pseudo is None or latent_b is None:
        raise RuntimeError("Multimap 模型未返回空间潜变量，无法执行迁移。")

    # 对齐尺寸
    if latent_b.shape != latent_a.shape:
        target_size = latent_b.shape[-2:]
        latent_a = F.interpolate(latent_a, size=target_size, mode="bilinear", align_corners=False)
        latent_a_pseudo = F.interpolate(latent_a_pseudo, size=target_size, mode="bilinear", align_corners=False)

    # 计算差异图并选择 Top-K 位置
    diff_map = compute_spatial_diff_map(latent_a, latent_a_pseudo, transfer_config.reduction)
    spatial_mask, top_positions = pick_top_spatial_positions(diff_map, transfer_config.top_percent)

    # 替换潜变量并解码
    updated_latent_b = replace_latent_positions(latent_a, latent_b, spatial_mask)

    with torch.no_grad():
        recon_modified = model.decode(updated_latent_b).clamp(0, 1)

    # 保存结果
    save_tensor_image(recon_modified.squeeze(0), str(output_paths.b_recon_modified))

    with open(output_paths.top_positions, "w") as f:
        if top_positions:
            f.write(";".join(f"{r},{c}" for r, c in top_positions[0]))

    logger.info(f"  Top {transfer_config.top_percent}% positions ({transfer_config.reduction})")
    logger.info(f"  Saved: {output_paths.b_recon_modified.name}")


# ============================================================================
# CLI Interface
# ============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将异常样本 latent 的显著空间位置拷贝到正常样本并解码重建。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入图像
    input_group = parser.add_argument_group("Input Images")
    input_group.add_argument("--abnormal", required=True, help="异常图像路径")
    input_group.add_argument("--normal", required=True, help="正常图像路径")

    # AOT-GAN 配置
    aot_group = parser.add_argument_group("AOT-GAN Config")
    aot_group.add_argument("--aae-c-ckpt", required=True, help="AAE_C (Ret_AAE) 权重")
    aot_group.add_argument("--aot-ckpt", required=True, help="AOT-GAN netG 权重")
    aot_group.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD, help="掩膜阈值")
    aot_group.add_argument("--color-mode", choices=["gray", "color"], default="gray", help="色彩模式")
    aot_group.add_argument("--aot-img-size", type=int, default=DEFAULT_AOT_IMG_SIZE, help="输入尺寸")

    # Multimap 配置
    mm_group = parser.add_argument_group("Multimap Config")
    mm_group.add_argument("--multimap-ckpt", required=True, help="AAE_S / Multimap 权重")
    mm_group.add_argument("--multimap-config", default=None, help="AAE_S 配置 JSON")
    mm_group.add_argument("--multimap-img-size", type=int, default=DEFAULT_MULTIMAP_IMG_SIZE, help="输入尺寸")

    # 潜变量迁移配置
    transfer_group = parser.add_argument_group("Latent Transfer Config")
    transfer_group.add_argument("--top-percent", type=float, default=DEFAULT_TOP_PERCENT,
                                help="复制的潜变量空间位置占比 (0-100)")
    transfer_group.add_argument("--latent-reduction", choices=["mean", "max"], default="mean",
                                help="差异图跨通道聚合方式")

    # 输出配置
    output_group = parser.add_argument_group("Output Config")
    output_group.add_argument("--out-dir", required=True, help="输出目录")
    output_group.add_argument("--device", default="cuda", help="计算设备")
    output_group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")

    return parser.parse_args()


def setup_device(device_str: str) -> torch.device:
    """设置计算设备。"""
    if device_str != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU。")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    """主入口函数。"""
    args = parse_args()

    # 初始化
    torch.manual_seed(args.seed)
    device = setup_device(args.device)

    # 准备输出目录
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    output_paths = OutputPaths(base_dir=out_dir)

    # 构建配置对象
    aot_config = AotGanConfig(
        aae_ckpt=Path(args.aae_c_ckpt),
        aot_ckpt=Path(args.aot_ckpt),
        mask_threshold=args.mask_threshold,
        color_mode=args.color_mode,
        img_size=args.aot_img_size,
    )

    multimap_config = MultimapConfig(
        ckpt_path=Path(args.multimap_ckpt),
        config_path=Path(args.multimap_config) if args.multimap_config else None,
        img_size=args.multimap_img_size,
    )

    transfer_config = LatentTransferConfig(
        top_percent=args.top_percent,
        reduction=args.latent_reduction,
    )

    # 执行流水线
    aot_out = step1_aotgan_inference(
        abnormal_path=Path(args.abnormal),
        config=aot_config,
        device=device,
        output_paths=output_paths,
    )

    out_a, out_a_pseudo, out_b, model = step2_multimap_inference(
        aot_out=aot_out,
        normal_path=Path(args.normal),
        multimap_config=multimap_config,
        device=device,
        output_paths=output_paths,
    )

    step3_latent_transfer(
        out_a=out_a,
        out_a_pseudo=out_a_pseudo,
        out_b=out_b,
        model=model,
        transfer_config=transfer_config,
        output_paths=output_paths,
    )

    logger.info(f"完成！输出目录: {out_dir}")


if __name__ == "__main__":
    main()
