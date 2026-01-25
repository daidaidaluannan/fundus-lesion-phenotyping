#!/usr/bin/env python3
"""
基于异常 latent 特征构建 Diffusion Pseudotime（DPT）进展轴：
1) 复用 AAE_C + AOT-GAN + AAE_S 提取 top-percent 异常区域 latent；
2) 在特征空间构建 kNN 图，Gaussian kernel 得到扩散转移矩阵；
3) 计算扩散映射坐标，选定根节点，得到伪时间（归一化到 [0,1]）；
4) 输出伪时间 CSV、扩散坐标 npz，以及 UMAP/t-SNE/扩散坐标散点可视化。
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import functional as TF
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# repo root 注入 sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from model_zoo.phanes import AnomalyMap  # noqa: E402
from transforms.synthetic import GenerateMasks  # noqa: E402
from train_demo import aotgan_1123 as aot  # noqa: E402
from train_demo.change_latent import (  # noqa: E402
    load_image as load_image_aot,
    compute_spatial_diff_map,
    pick_top_spatial_positions,
)
from train_demo.visualize_multimap_single_1113 import (  # noqa: E402
    ModelConfig,
    build_multimap_model,
    ensure_dir,
    save_tensor_image,
)


ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


@dataclass
class SampleRecord:
    name: str
    source_path: Path
    feature: np.ndarray
    anomaly_score: float
    preview_path: Path
    heatmap_path: Path
    overlay_path: Path


@dataclass
class PipelineContext:
    device: torch.device
    multimap_model: torch.nn.Module
    ret_aae: torch.nn.Module
    netG: torch.nn.Module
    mask_generator: GenerateMasks
    anomaly_map: AnomalyMap
    color_mode: str
    mask_threshold: float
    aot_img_size: int
    multimap_img_size: int
    top_percent: float
    latent_reduction: str
    feature_agg: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 latent 异常特征的 Diffusion Pseudotime 进展轴。")
    parser.add_argument("--data-dir", required=True, help="图像根目录")
    parser.add_argument("--csv", required=True, help="包含图像名称的 CSV，首列为文件名/ID")
    parser.add_argument("--out-dir", required=True, help="输出目录")

    parser.add_argument("--aae-c-ckpt", required=True, help="AAE_C (Ret_AAE) 权重路径")
    parser.add_argument("--aot-ckpt", required=True, help="AOT-GAN netG 权重路径")
    parser.add_argument("--mask-threshold", type=float, default=0.153, help="AOT-GAN 掩膜阈值")
    parser.add_argument("--color-mode", choices=["gray", "color"], default="gray", help="AOT-GAN 训练色彩模式")
    parser.add_argument("--aot-img-size", type=int, default=224, help="AOT-GAN 输入尺寸")

    parser.add_argument("--multimap-ckpt", required=True, help="AAE_S / multimap 权重")
    parser.add_argument("--multimap-config", default=None, help="AAE_S 配置 JSON")
    parser.add_argument("--multimap-img-size", type=int, default=224, help="AAE_S 输入尺寸")

    parser.add_argument("--top-percent", type=float, default=10.0, help="latent 差异中选取的空间位置百分比")
    parser.add_argument("--latent-reduction", choices=["mean", "max"], default="mean", help="跨通道聚合方式")
    parser.add_argument("--feature-agg",
                        choices=["flatten", "channel-mean", "channel-max", "spatial-mean", "spatial-max"],
                        default="flatten",
                        help=("flatten: masked latent 展平(128xHxW)；"
                              "channel-mean/channel-max: 通道聚合 128 维；"
                              "spatial-mean/spatial-max: 空间展平"))

    parser.add_argument("--knn-k", type=int, default=15, help="kNN 邻居数")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean", help="kNN 距离度量")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian kernel σ；默认取 kNN 距离的中位数")
    parser.add_argument("--n-eigs", type=int, default=5, help="保留的非平凡扩散特征向量数")

    parser.add_argument("--root-name", default=None, help="指定根节点样本名（stem）")
    parser.add_argument("--root-index", type=int, default=None, help="指定根节点索引（0-based）")
    parser.add_argument("--root-strategy", choices=["min-anomaly", "first"], default="min-anomaly",
                        help="未指定 root 时的默认策略")

    parser.add_argument("--pseudotime-mode",
                        choices=["dpt", "mst", "both"],
                        default="dpt",
                        help="选择伪时间算法：DPT、MST 或同时输出")
    parser.add_argument("--mst-space",
                        choices=["diffusion", "feature"],
                        default="diffusion",
                        help="MST 距离计算空间：扩散坐标或原始特征")
    parser.add_argument("--mst-metric",
                        choices=["euclidean", "cosine"],
                        default="euclidean",
                        help="MST 距离度量（仅在 MST 模式时使用）")

    parser.add_argument("--enable-umap", action="store_true", help="启用 UMAP 可视化")
    parser.add_argument("--enable-tsne", action="store_true", help="启用 t-SNE 可视化")

    parser.add_argument("--device", default="cuda", help="运行设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def read_image_list(csv_path: Path) -> List[str]:
    names: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if name and name not in names:
                names.append(name)
    return names


def build_image_index(data_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    index_by_name: Dict[str, Path] = {}
    index_by_stem: Dict[str, List[Path]] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTS:
            continue
        name = path.name
        stem = path.stem
        if name not in index_by_name:
            index_by_name[name] = path
        index_by_stem.setdefault(stem, []).append(path)
    return index_by_name, index_by_stem


def find_image_path(name: str,
                    data_dir: Path,
                    index_by_name: Dict[str, Path],
                    index_by_stem: Dict[str, List[Path]]) -> Optional[Path]:
    candidate = data_dir / name
    if candidate.exists():
        return candidate
    if name in index_by_name:
        return index_by_name[name]
    stem = Path(name).stem
    for ext in ALLOWED_EXTS:
        path = data_dir / f"{stem}{ext}"
        if path.exists():
            return path
    if stem in index_by_stem and index_by_stem[stem]:
        return index_by_stem[stem][0]
    return None


def _normalize_heatmap(arr: np.ndarray,
                       vmin_pct: float = 1.0,
                       vmax_pct: float = 99.0,
                       gamma: float = 0.8) -> np.ndarray:
    flat = arr.reshape(-1)
    vmin = np.percentile(flat, vmin_pct)
    vmax = np.percentile(flat, vmax_pct)
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-8)
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    return arr


def save_heatmap(diff_map: torch.Tensor, out_path: Path) -> None:
    arr = diff_map.squeeze(0).detach().cpu()
    arr = arr - arr.min()
    denom = arr.max().clamp(min=1e-8)
    arr = (arr / denom).numpy()
    arr = _normalize_heatmap(arr, vmin_pct=1.0, vmax_pct=99.0, gamma=0.8)
    cmap = cm.get_cmap("jet")
    colored = (cmap(arr)[..., :3] * 255).astype(np.uint8)
    plt.imsave(out_path, colored)


def create_overlay_heatmap(original: torch.Tensor,
                           diff_map: torch.Tensor,
                           out_path: Path,
                           alpha: float = 0.5,
                           cmap_name: str = "jet") -> None:
    if original.ndim == 4:
        original = original.squeeze(0)
    if original.size(0) == 1:
        original = original.repeat(3, 1, 1)

    if diff_map.ndim == 2:
        diff_map = diff_map.unsqueeze(0).unsqueeze(0)
    elif diff_map.ndim == 3:
        diff_map = diff_map.unsqueeze(1)
    elif diff_map.ndim != 4:
        raise ValueError("diff_map 维度应为 2/3/4")

    diff_resized = F.interpolate(
        diff_map,
        size=original.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    diff_norm = diff_resized - diff_resized.min()
    diff_norm = diff_norm / diff_norm.max().clamp(min=1e-8)

    heat_map = diff_norm.detach().cpu().numpy()
    heat_map = _normalize_heatmap(heat_map, vmin_pct=1.0, vmax_pct=99.0, gamma=0.8)
    orig_np = original.detach().cpu().permute(1, 2, 0).numpy()
    orig_uint8 = (orig_np * 255.0).clip(0, 255).astype(np.uint8)

    colored = cm.get_cmap(cmap_name)(heat_map)[..., :3]
    colored_uint8 = (colored * 255.0).astype(np.uint8)

    overlay = (1.0 - alpha) * orig_uint8.astype(np.float32) + alpha * colored_uint8.astype(np.float32)
    overlay_uint8 = overlay.clip(0, 255).astype(np.uint8)
    plt.imsave(out_path, overlay_uint8)


@torch.no_grad()
def run_aotgan_preloaded(
    image_tensor: torch.Tensor,
    ctx: PipelineContext,
) -> Dict[str, torch.Tensor]:
    coarse_rgb, _ = ctx.ret_aae(image_tensor)
    x_gray = TF.rgb_to_grayscale(image_tensor)
    coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
    mask = aot.generate_mask(
        ctx.anomaly_map,
        ctx.mask_generator,
        x_gray,
        coarse_gray,
        mask_threshold=ctx.mask_threshold,
        add_synthetic=False)

    if ctx.color_mode == "color":
        mask_rgb = mask.repeat(1, 3, 1, 1)
        transformed = (image_tensor * (1 - mask_rgb)) + mask_rgb
    else:
        mask_rgb = mask
        transformed = (x_gray * (1 - mask)) + mask

    pred = ctx.netG(transformed, mask)
    if ctx.color_mode == "gray":
        pred = pred.repeat(1, 3, 1, 1)
    pred = pred.clamp(0.0, 1.0)

    return {
        "input": image_tensor,
        "coarse": coarse_rgb.clamp(0, 1),
        "mask": mask,
        "mask_rgb": mask_rgb,
        "pseudo": pred,
    }


@torch.no_grad()
def extract_sample(image_path: Path, ctx: PipelineContext, out_dir: Path) -> SampleRecord:
    img_aot = load_image_aot(image_path, ctx.aot_img_size, ctx.device)
    aot_out = run_aotgan_preloaded(img_aot, ctx)

    img_viz = F.interpolate(
        aot_out["input"],
        size=(ctx.multimap_img_size, ctx.multimap_img_size),
        mode="bilinear",
        align_corners=False,
    )
    img_pseudo = F.interpolate(
        aot_out["pseudo"],
        size=(ctx.multimap_img_size, ctx.multimap_img_size),
        mode="bilinear",
        align_corners=False,
    )

    out_a = ctx.multimap_model(img_viz)
    out_pseudo = ctx.multimap_model(img_pseudo)

    if isinstance(out_a, (list, tuple)):
        if len(out_a) == 2:
            recon_a, latent_a = out_a
        elif len(out_a) == 3:
            recon_a, _, latent_a = out_a
        else:
            raise RuntimeError(f"multimap 输出长度不支持: {len(out_a)}")
    else:
        raise RuntimeError("multimap 模型未返回 latent。")

    if isinstance(out_pseudo, (list, tuple)):
        if len(out_pseudo) == 2:
            recon_pseudo, latent_pseudo = out_pseudo
        elif len(out_pseudo) == 3:
            recon_pseudo, _, latent_pseudo = out_pseudo
        else:
            raise RuntimeError(f"multimap 输出长度不支持: {len(out_pseudo)}")
    else:
        raise RuntimeError("multimap 模型未返回 latent。")

    recon_a = recon_a.clamp(0, 1)
    recon_pseudo = recon_pseudo.clamp(0, 1)

    if latent_a.shape != latent_pseudo.shape:
        latent_pseudo = F.interpolate(latent_pseudo, size=latent_a.shape[-2:], mode="bilinear", align_corners=False)

    diff_map = compute_spatial_diff_map(latent_a, latent_pseudo, ctx.latent_reduction)  # B,H,W
    anomaly_score = float(diff_map.mean().item())

    spatial_mask, _ = pick_top_spatial_positions(diff_map, ctx.top_percent)
    masked_latent = latent_a * spatial_mask
    if ctx.feature_agg == "flatten":
        feature_vec = masked_latent.flatten(start_dim=1).squeeze(0).detach().cpu().numpy()
    elif ctx.feature_agg == "channel-mean":
        area = spatial_mask.sum(dim=(2, 3)).clamp(min=1e-6)
        feature_vec = (masked_latent.sum(dim=(2, 3)) / area).squeeze(0).detach().cpu().numpy()
    elif ctx.feature_agg == "channel-max":
        masked_for_max = torch.where(spatial_mask.bool(), latent_a, torch.full_like(latent_a, float("-inf")))
        channel_max = masked_for_max.amax(dim=(2, 3))
        channel_max = torch.where(torch.isfinite(channel_max), channel_max, torch.zeros_like(channel_max))
        feature_vec = channel_max.squeeze(0).detach().cpu().numpy()
    elif ctx.feature_agg == "spatial-mean":
        spatial_mean = masked_latent.mean(dim=1)
        feature_vec = spatial_mean.flatten(start_dim=1).squeeze(0).detach().cpu().numpy()
    elif ctx.feature_agg == "spatial-max":
        masked_for_max = torch.where(spatial_mask.bool(), latent_a, torch.full_like(latent_a, float("-inf")))
        spatial_max = masked_for_max.amax(dim=1)
        spatial_max = torch.where(torch.isfinite(spatial_max), spatial_max, torch.zeros_like(spatial_max))
        feature_vec = spatial_max.flatten(start_dim=1).squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(f"未知 feature_agg: {ctx.feature_agg}")

    feature_source = diff_map.unsqueeze(1)

    indiv_dir = out_dir / "individuals"
    ensure_dir(indiv_dir)
    stem = image_path.stem
    preview_path = indiv_dir / f"{stem}_input.png"
    save_tensor_image(img_viz.squeeze(0), str(preview_path))

    heatmap_path = indiv_dir / f"{stem}_heatmap.png"
    save_heatmap(feature_source.squeeze(0), heatmap_path)
    overlay_path = indiv_dir / f"{stem}_overlay.png"
    create_overlay_heatmap(img_viz, feature_source.squeeze(1), overlay_path)

    return SampleRecord(
        name=stem,
        source_path=image_path,
        feature=feature_vec,
        anomaly_score=anomaly_score,
        preview_path=preview_path,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )


def collect_samples(image_paths: List[Path], ctx: PipelineContext, out_dir: Path) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    for idx, path in enumerate(image_paths):
        try:
            print(f"[{idx+1}/{len(image_paths)}] 处理 {path.name}")
            record = extract_sample(path, ctx, out_dir)
            samples.append(record)
        except Exception as e:
            print(f"[warn] 处理 {path} 失败：{e}")
    return samples


def build_knn_graph(features: np.ndarray,
                    k: int,
                    metric: str,
                    sigma: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """返回对称邻接矩阵 W 和度向量 d。"""
    n = features.shape[0]
    k = int(max(2, min(k, n - 1)))
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)

    if sigma is None:
        # 使用全部 kNN 距离（排除对角）中位数，避免 σ 过小导致全零。
        sigma = float(np.median(distances[:, 1:].reshape(-1)) + 1e-8)
    sigma_sq = sigma ** 2

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = np.exp(- (float(dist) ** 2) / sigma_sq)
            if sim <= 0:
                continue
            # 对称化取最大值，避免双向重复
            if sim > W[i, j]:
                W[i, j] = sim
                W[j, i] = sim
    d = W.sum(axis=1)
    return W, d


def diffusion_map(W: np.ndarray,
                  d: np.ndarray,
                  n_eigs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (eigvals, diffusion_coords)，跳过最大的平凡特征值。
    使用对称化的 K = D^{-1/2} W D^{-1/2} 做特征分解。
    """
    eps = 1e-8
    d_safe = d + eps
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    K = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 跳过平凡特征向量（对应最大特征值 ~1）
    eigvals_nontrivial = eigvals[1:n_eigs + 1]
    eigvecs_nontrivial = eigvecs[:, 1:n_eigs + 1]
    psi = eigvecs_nontrivial * d_inv_sqrt[:, None]
    diffusion_coords = psi * eigvals_nontrivial  # 每个坐标缩放特征值
    return eigvals_nontrivial, diffusion_coords


def choose_root(samples: List[SampleRecord],
                root_name: Optional[str],
                root_index: Optional[int],
                strategy: str) -> int:
    if root_name is not None:
        for idx, s in enumerate(samples):
            if s.name == root_name:
                return idx
        raise ValueError(f"未找到指定 root-name: {root_name}")
    if root_index is not None:
        if 0 <= root_index < len(samples):
            return root_index
        raise ValueError(f"root-index 越界: {root_index}")
    if strategy == "first":
        return 0
    # 默认：异常分数最小视为最健康
    return int(np.argmin([s.anomaly_score for s in samples]))


def compute_pseudotime(diff_coords: np.ndarray,
                       anomaly_scores: np.ndarray,
                       root_idx: int) -> np.ndarray:
    if diff_coords.ndim != 2 or diff_coords.shape[0] == 0:
        raise ValueError("扩散坐标为空，无法计算伪时间。")
    root_coord = diff_coords[root_idx]
    dists = np.linalg.norm(diff_coords - root_coord[None, :], axis=1)
    d_min, d_max = dists.min(), dists.max()
    if d_max - d_min < 1e-12:
        pt = np.zeros_like(dists)
    else:
        pt = (dists - d_min) / (d_max - d_min)
    # 方向校准：与异常分数正相关
    if anomaly_scores.size > 1:
        corr = np.corrcoef(pt, anomaly_scores)[0, 1]
        if np.isfinite(corr) and corr < 0:
            pt = 1.0 - pt
    pt[root_idx] = 0.0
    return pt


def save_pseudotime_csv(out_path: Path,
                        samples: List[SampleRecord],
                        pseudotime_dpt: Optional[np.ndarray],
                        pseudotime_mst: Optional[np.ndarray],
                        root_idx: int) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "source", "anomaly_score", "pseudotime_dpt", "pseudotime_mst", "is_root"])
        for i, s in enumerate(samples):
            pt_dpt = float(pseudotime_dpt[i]) if pseudotime_dpt is not None else ""
            pt_mst = float(pseudotime_mst[i]) if pseudotime_mst is not None else ""
            writer.writerow([s.name, str(s.source_path), s.anomaly_score, pt_dpt, pt_mst, int(i == root_idx)])


def plot_diffusion_scatter(diff_coords: np.ndarray,
                           pseudotime: np.ndarray,
                           out_path: Path,
                           title: str = "Diffusion components") -> None:
    if diff_coords.shape[1] < 2:
        print("[DPT] 扩散坐标维度不足 2，跳过散点图。")
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(diff_coords[:, 0], diff_coords[:, 1], c=pseudotime, cmap="viridis", s=30, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, shrink=0.8, label="pseudotime")
    plt.xlabel("DC1")
    plt.ylabel("DC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pairwise_distance(mat: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        diff = mat[:, None, :] - mat[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    elif metric == "cosine":
        norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        normed = mat / norm
        sim = normed @ normed.T
        dist = 1.0 - sim
        dist = np.clip(dist, 0.0, 2.0)
    else:
        raise ValueError(f"未知距离度量: {metric}")
    np.fill_diagonal(dist, 0.0)
    return dist


def build_mst(dist: np.ndarray) -> np.ndarray:
    """
    用 Prim 算法构建 MST，返回父节点数组 parents，其中 parents[root]= -1。
    dist 需为对称非负矩阵。
    """
    n = dist.shape[0]
    if n < 2:
        return np.array([-1], dtype=int)
    parents = -np.ones(n, dtype=int)
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    root = 0
    key[root] = 0.0
    for _ in range(n):
        u = int(np.argmin(key))
        in_mst[u] = True
        key[u] = np.inf
        for v in range(n):
            if not in_mst[v] and dist[u, v] < key[v]:
                key[v] = dist[u, v]
                parents[v] = u
    return parents


def mst_pseudotime(dist: np.ndarray, parents: np.ndarray, root_idx: int) -> np.ndarray:
    """
    在已构建的 MST 上，从 root 出发累积边权得到伪时间，并归一化到 [0,1]。
    """
    n = dist.shape[0]
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for v in range(n):
        u = parents[v]
        if u >= 0:
            w = float(dist[u, v])
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))

    pt = np.full(n, np.inf, dtype=float)
    pt[root_idx] = 0.0
    queue = [root_idx]
    while queue:
        curr = queue.pop(0)
        for nxt, w in adjacency[curr]:
            if pt[nxt] == np.inf:
                pt[nxt] = pt[curr] + w
                queue.append(nxt)
    # 归一化
    finite_mask = np.isfinite(pt)
    if finite_mask.sum() == 0:
        return np.zeros(n, dtype=float)
    d_min = pt[finite_mask].min()
    d_max = pt[finite_mask].max()
    if d_max - d_min < 1e-12:
        pt_norm = np.zeros_like(pt)
    else:
        pt_norm = (pt - d_min) / (d_max - d_min)
    pt_norm[~finite_mask] = 0.0
    return pt_norm


def save_umap(features: np.ndarray, pseudotime: np.ndarray, out_path: Path, seed: int) -> None:
    if not HAS_UMAP:
        print("[UMAP] 未安装 umap-learn，跳过 UMAP。")
        return
    reducer = umap.UMAP(random_state=seed)
    embed = reducer.fit_transform(features)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(embed[:, 0], embed[:, 1], c=pseudotime, cmap="viridis", s=30, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, shrink=0.8, label="pseudotime")
    plt.title("UMAP (colored by pseudotime)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_tsne(features: np.ndarray, pseudotime: np.ndarray, out_path: Path, seed: int) -> None:
    if not HAS_TSNE:
        print("[t-SNE] sklearn.manifold.TSNE 不可用，跳过。")
        return
    if features.shape[0] < 2:
        print("[t-SNE] 样本不足，跳过。")
        return
    perplexity = min(30, max(2, features.shape[0] - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embed = reducer.fit_transform(features)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(embed[:, 0], embed[:, 1], c=pseudotime, cmap="viridis", s=30, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, shrink=0.8, label="pseudotime")
    plt.title("t-SNE (colored by pseudotime)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("[Pseudotime] CUDA 不可用，改用 CPU。")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    data_dir = Path(args.data_dir)
    print("[Pseudotime] 扫描图像索引中...")
    index_by_name, index_by_stem = build_image_index(data_dir)
    print(f"[Pseudotime] 索引完成：name 唯一键 {len(index_by_name)}，stem 组 {len(index_by_stem)}")

    names = read_image_list(Path(args.csv))
    image_paths: List[Path] = []
    for name in names:
        path = find_image_path(name, data_dir, index_by_name, index_by_stem)
        if path:
            image_paths.append(path)
        else:
            print(f"[warn] 未找到图像：{name}")
    if len(image_paths) < 2:
        print("[Pseudotime] 可处理图像不足 2 张，退出。")
        return

    # 构建模型
    multimap_model, _ = build_multimap_model(
        ModelConfig(
            ckpt_path=args.multimap_ckpt,
            config_path=args.multimap_config,
        ),
        device=device,
    )
    ret_aae = aot.load_ret_aae(Path(args.aae_c_ckpt), device)
    netG, _ = aot.build_aotgan_networks(args.color_mode, device)
    ckpt = torch.load(args.aot_ckpt, map_location=device)
    state = ckpt.get("netG", ckpt)
    netG.load_state_dict(state)
    netG.eval()

    ctx = PipelineContext(
        device=device,
        multimap_model=multimap_model,
        ret_aae=ret_aae,
        netG=netG,
        mask_generator=GenerateMasks(min_size=20, max_size=40),
        anomaly_map=AnomalyMap(),
        color_mode=args.color_mode,
        mask_threshold=args.mask_threshold,
        aot_img_size=args.aot_img_size,
        multimap_img_size=args.multimap_img_size,
        top_percent=args.top_percent,
        latent_reduction=args.latent_reduction,
        feature_agg=args.feature_agg,
        seed=args.seed,
    )

    samples = collect_samples(image_paths, ctx, out_dir)
    if len(samples) < 2:
        print("[Pseudotime] 样本不足以计算伪时间。")
        return

    features = np.stack([s.feature for s in samples], axis=0)
    anomaly_scores = np.array([s.anomaly_score for s in samples], dtype=np.float64)
    np.savez(out_dir / "features.npz", features=features, names=[s.name for s in samples])

    W, d = build_knn_graph(features, args.knn_k, args.metric, args.sigma)
    eigvals, diff_coords = diffusion_map(W, d, args.n_eigs)

    try:
        root_idx = choose_root(samples, args.root_name, args.root_index, args.root_strategy)
    except Exception as e:
        print(f"[Pseudotime] 选择根节点失败：{e}")
        return

    pseudotime_dpt: Optional[np.ndarray] = None
    pseudotime_mst: Optional[np.ndarray] = None

    if args.pseudotime_mode in ("dpt", "both"):
        pseudotime_dpt = compute_pseudotime(diff_coords, anomaly_scores, root_idx)

    if args.pseudotime_mode in ("mst", "both"):
        if args.mst_space == "diffusion":
            base = diff_coords
        else:
            base = features
        dist_mat = pairwise_distance(base, args.mst_metric)
        parents = build_mst(dist_mat)
        pseudotime_mst = mst_pseudotime(dist_mat, parents, root_idx)
        # 方向校准：与异常分数正相关
        if anomaly_scores.size > 1:
            corr = np.corrcoef(pseudotime_mst, anomaly_scores)[0, 1]
            if np.isfinite(corr) and corr < 0:
                pseudotime_mst = 1.0 - pseudotime_mst
        pseudotime_mst[root_idx] = 0.0

    # 保存结果
    np.savez(out_dir / "diffusion_coords.npz",
             eigvals=eigvals,
             diffusion_coords=diff_coords,
             pseudotime_dpt=pseudotime_dpt,
             pseudotime_mst=pseudotime_mst,
             names=[s.name for s in samples])
    save_pseudotime_csv(out_dir / "pseudotime.csv", samples, pseudotime_dpt, pseudotime_mst, root_idx)

    if pseudotime_dpt is not None:
        plot_diffusion_scatter(diff_coords, pseudotime_dpt, out_dir / "diffusion_scatter.png")
    if pseudotime_mst is not None:
        plot_diffusion_scatter(diff_coords, pseudotime_mst, out_dir / "diffusion_scatter_mst.png", title="Diffusion components (MST PT)")

    if args.enable_umap:
        if pseudotime_dpt is not None:
            save_umap(features, pseudotime_dpt, out_dir / "umap_pseudotime.png", seed=args.seed)
        if pseudotime_mst is not None:
            save_umap(features, pseudotime_mst, out_dir / "umap_pseudotime_mst.png", seed=args.seed)
    if args.enable_tsne:
        if pseudotime_dpt is not None:
            save_tsne(features, pseudotime_dpt, out_dir / "tsne_pseudotime.png", seed=args.seed)
        if pseudotime_mst is not None:
            save_tsne(features, pseudotime_mst, out_dir / "tsne_pseudotime_mst.png", seed=args.seed)

    print("[Pseudotime] 完成。输出目录：", out_dir)


if __name__ == "__main__":
    main()
