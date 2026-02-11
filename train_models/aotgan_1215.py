#!/usr/bin/env python
"""
Extended AOT-GAN training demo that keeps the original behavior while adding
dataset utilities, multi-GPU training, resumable checkpoints, and logging.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parent.parent
import sys
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from model_zoo.aotgan.aotgan import InpaintGenerator, Discriminator
from model_zoo.aotgan.loss import loss as loss_module
from train_demo.AAE_C  import AAE


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, img_size: int = 224, mask_dir: Optional[Path] = None):
        self.image_paths = sorted([
            image_dir / fname for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        self.mask_paths: Optional[List[Path]] = None
        if mask_dir is not None:
            self.mask_paths = []
            missing = []
            for img_path in self.image_paths:
                m_path = mask_dir / img_path.name
                if not m_path.is_file():
                    missing.append(img_path.name)
                self.mask_paths.append(m_path)
            if missing:
                print(f"[Warn] {len(missing)} masks not found in {mask_dir}. They will be generated if enabled.")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [0,1]
        ])
        self.cached_images: Optional[torch.Tensor] = None
        self.cached_masks: Optional[torch.Tensor] = None
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def set_cache(self, images: torch.Tensor, masks: Optional[torch.Tensor]):
        self.cached_images = images
        self.cached_masks = masks

    def __getitem__(self, idx):
        if self.cached_images is not None:
            img = self.cached_images[idx]
            if self.cached_masks is not None:
                return img, self.cached_masks[idx]
            return img
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_t = self.transform(img)
        if self.cached_masks is not None:
            return img_t, self.cached_masks[idx]
        if self.mask_paths is not None:
            mask_img = Image.open(self.mask_paths[idx]).convert("L")
            mask_img = TF.resize(mask_img, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
            mask_t = TF.to_tensor(mask_img)
            mask_t = (mask_t > 0.5).float()
            return img_t, mask_t
        return img_t


class RetinaDataset(Dataset):
    """Dataset helper (ported from AAE_S_1206) for image/mask pairs with augments."""

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        train: bool = True,
        grayscale: bool = False,
        img_size: int = 224,
        apply_color_jitter: bool = False,
    ):
        assert mask_paths is None or len(mask_paths) == len(image_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.train = train
        self.grayscale = grayscale
        self.img_size = img_size
        self.apply_color_jitter = apply_color_jitter
        self.cj = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if apply_color_jitter else None
        self.cached_images: Optional[torch.Tensor] = None
        self.cached_masks: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.image_paths)

    def set_cache(self, images: torch.Tensor, masks: Optional[torch.Tensor]):
        self.cached_images = images
        self.cached_masks = masks

    def _load_pair(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("L" if self.grayscale else "RGB")
        m = None
        if self.mask_paths is not None:
            m = Image.open(self.mask_paths[idx]).convert("L")
        return img, m

    def __getitem__(self, idx: int):
        if self.cached_images is not None:
            img_t = self.cached_images[idx]
            mask_cached = self.cached_masks[idx] if self.cached_masks is not None else None
            return img_t, mask_cached

        img, m = self._load_pair(idx)
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        if m is not None:
            m = TF.resize(m, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                if m is not None:
                    m = TF.hflip(m)
            angle = random.uniform(-15.0, 15.0)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            if m is not None:
                m = TF.rotate(m, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            if self.cj is not None:
                img = self.cj(img)

        img_t = TF.to_tensor(img)
        mask_t = None
        if m is not None:
            mask_t = TF.to_tensor(m)
            mask_t = (mask_t > 0.5).float()
        if self.cached_masks is not None:
            mask_t = self.cached_masks[idx]
        return img_t, mask_t


def collate_with_optional_masks(batch):
    imgs, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    if all(m is None for m in masks):
        return imgs, None
    masks = torch.stack([m for m in masks], dim=0)
    return imgs, masks


def load_image_labels(
    csv_path: str,
    image_column: str = "image_name",
    status_column: str = "status_label",
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or image_column not in reader.fieldnames or status_column not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must contain columns '{image_column}' and '{status_column}'")
        for row in reader:
            name = row.get(image_column)
            label = row.get(status_column)
            if not name or label is None:
                continue
            labels[name.strip()] = str(label).strip()
    return labels


def split_dataset_paths(
    image_paths: List[str],
    mask_paths: Optional[List[str]],
    train_ratio: float,
    seed: int,
    labels: Optional[Dict[str, str]] = None,
) -> Tuple[Tuple[List[str], Optional[List[str]]], Tuple[List[str], Optional[List[str]]]]:
    if not image_paths:
        raise ValueError("No images available for splitting")
    if mask_paths is not None and len(mask_paths) != len(image_paths):
        raise ValueError("Image/mask count mismatch when splitting")

    total = len(image_paths)
    if total < 2:
        raise ValueError("Need at least 2 samples to create train/val split")

    ratio = float(train_ratio)
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

    rng = random.Random(seed)

    def gather(idxs: List[int]) -> Tuple[List[str], Optional[List[str]]]:
        imgs = [image_paths[i] for i in idxs]
        masks = [mask_paths[i] for i in idxs] if mask_paths is not None else None
        return imgs, masks

    if not labels:
        indices = list(range(total))
        rng.shuffle(indices)
        split_idx = max(1, min(total - 1, int(round(total * ratio))))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return gather(train_idx), gather(val_idx)

    label_groups: Dict[str, List[int]] = {}
    unlabeled: List[int] = []
    for idx, path in enumerate(image_paths):
        name = os.path.basename(path)
        label = labels.get(name)
        if label is None:
            unlabeled.append(idx)
            continue
        label_groups.setdefault(label, []).append(idx)

    if not label_groups:
        indices = list(range(total))
        rng.shuffle(indices)
        split_idx = max(1, min(total - 1, int(round(total * ratio))))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return gather(train_idx), gather(val_idx)

    def split_bucket(bucket: List[int]) -> Tuple[List[int], List[int]]:
        n = len(bucket)
        if n <= 1:
            return bucket.copy(), []
        split_idx = int(round(n * ratio))
        split_idx = max(1, min(n - 1, split_idx))
        return bucket[:split_idx], bucket[split_idx:]

    train_idx: List[int] = []
    val_idx: List[int] = []
    for bucket in label_groups.values():
        rng.shuffle(bucket)
        t_idx, v_idx = split_bucket(bucket)
        train_idx.extend(t_idx)
        val_idx.extend(v_idx)

    if unlabeled:
        rng.shuffle(unlabeled)
        t_idx, v_idx = split_bucket(unlabeled)
        train_idx.extend(t_idx)
        val_idx.extend(v_idx)

    if not val_idx:
        if train_idx:
            val_idx.append(train_idx.pop())
        else:
            raise RuntimeError("Unable to create validation split; not enough labeled samples.")

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return gather(train_idx), gather(val_idx)


def resolve_image_mask_paths(
    image_source: Union[str, Sequence[str]],
    mask_source: Optional[Union[str, Sequence[str]]],
    allowed_names: Optional[Set[str]] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def is_sequence(obj):
        return isinstance(obj, (list, tuple))

    if is_sequence(image_source):
        img_paths = list(image_source)
        base_names = [os.path.basename(p) for p in img_paths]
    else:
        files = [f for f in sorted(os.listdir(image_source)) if f.lower().endswith(exts)]
        if allowed_names is not None:
            files = [f for f in files if f in allowed_names]
        img_paths = [os.path.join(image_source, f) for f in files]
        base_names = files

    if mask_source is None:
        mask_paths = None
    elif is_sequence(mask_source):
        mask_paths = list(mask_source)
        if len(mask_paths) != len(img_paths):
            raise ValueError("Mask path count must match image path count")
    else:
        mask_candidates = {f: os.path.join(mask_source, f) for f in os.listdir(mask_source) if f.lower().endswith(exts)}
        mask_paths = []
        for name in base_names:
            path = mask_candidates.get(name)
            if path is None:
                raise FileNotFoundError(f"Mask for {name} not found in {mask_source}")
            mask_paths.append(path)

    return img_paths, mask_paths


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def gather_vis_samples(dataset, max_samples: int = 4) -> torch.Tensor:
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot gather visualization samples")
    samples = []
    count = min(max_samples, len(dataset))
    for idx in range(count):
        item = dataset[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item
        samples.append(img)
    return torch.stack(samples, dim=0)


def split_batch(batch):
    """Return (images, masks) regardless of collate style."""
    if isinstance(batch, torch.Tensor):
        return batch, None
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2 and isinstance(batch[0], torch.Tensor):
            return batch[0], batch[1]
        if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0), None
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def compute_mask_gpu(x_gray: torch.Tensor,
                     coarse_gray: torch.Tensor,
                     mask_threshold: Optional[float]) -> torch.Tensor:
    residual = torch.abs(coarse_gray - x_gray)
    residual = F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
    # Normalize by per-sample 95th percentile to stabilize scale
    q = torch.quantile(residual.flatten(1), 0.95, dim=1, keepdim=True)
    residual = residual / (q.view(-1, 1, 1, 1) + 1e-6)
    if mask_threshold is None:
        thresh = torch.quantile(residual.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    else:
        thresh = torch.full((residual.size(0), 1, 1, 1), float(mask_threshold), device=residual.device)
    mask = (residual > thresh).float()
    mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return torch.clamp(mask, 0.0, 1.0)


@torch.no_grad()
def generate_masks_to_dir(dataset: ImageFolderDataset,
                          ret_aae,
                          device: torch.device,
                          mask_threshold: float,
                          batch_size: int,
                          num_workers: int,
                          mask_dir: Path,
                          progress: bool = False):
    mask_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    iterator = tqdm(loader, desc="Generate masks", leave=False) if progress else loader
    saved = 0
    skipped = 0
    processed = 0
    ret_aae.eval()
    for batch in iterator:
        imgs, _ = split_batch(batch)
        imgs = imgs.to(device)
        coarse_rgb, _ = ret_aae(imgs)
        x_gray = TF.rgb_to_grayscale(imgs)
        coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
        masks = compute_mask_gpu(x_gray, coarse_gray, mask_threshold)
        batch_paths = dataset.image_paths[processed: processed + imgs.size(0)]
        for i, img_path in enumerate(batch_paths):
            mask_path = mask_dir / Path(img_path).name
            if mask_path.exists():
                skipped += 1
                continue
            save_image(masks[i].cpu(), mask_path)
            saved += 1
        processed += imgs.size(0)
    print(f"[Mask] Generated {saved} masks (skipped {skipped} existing) -> {mask_dir}")


@torch.no_grad()
def precompute_and_cache_masks(dataset,
                               ret_aae,
                               device: torch.device,
                               mask_threshold: Optional[float],
                               batch_size: int,
                               num_workers: int,
                               progress: bool = False):
    if dataset is None:
        return
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_with_optional_masks if isinstance(dataset, RetinaDataset) else None,
        drop_last=False,
    )
    cached_imgs: List[torch.Tensor] = []
    cached_masks: List[torch.Tensor] = []
    iterator = tqdm(loader, desc="Precompute masks", leave=False) if progress else loader
    ret_aae.eval()
    for batch in iterator:
        imgs, provided_mask = split_batch(batch)
        imgs = imgs.to(device)
        if provided_mask is not None:
            mask = provided_mask.to(device)
        else:
            coarse_rgb, _ = ret_aae(imgs)
            x_gray = TF.rgb_to_grayscale(imgs)
            coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
            mask = compute_mask_gpu(x_gray, coarse_gray, mask_threshold)
        cached_imgs.append(imgs.cpu())
        cached_masks.append(mask.cpu())
    images_tensor = torch.cat(cached_imgs, dim=0) if cached_imgs else None
    masks_tensor = torch.cat(cached_masks, dim=0) if cached_masks else None
    if hasattr(dataset, "set_cache"):
        dataset.set_cache(images_tensor, masks_tensor)
    else:
        raise AttributeError("Dataset does not support caching masks.")


def prepare_dataloaders(args, drop_last_train: bool = False, drop_last_val: bool = False):
    def limit_paths(imgs: List[str], masks: Optional[List[str]], limit: Optional[int], split_name: str):
        if limit is None or limit <= 0:
            return imgs, masks
        n = min(limit, len(imgs))
        if n <= 0:
            return imgs, masks
        if masks is not None:
            masks = masks[:n]
        print(f"[Info] 限制{split_name}样本数为 {n} (原始 {len(imgs)})")
        return imgs[:n], masks

    if args.train_images is None:
        if args.image_dir is None:
            raise ValueError("Either --image-dir or --train-images must be provided")
        dataset = ImageFolderDataset(Path(args.image_dir), img_size=args.img_size)
        if args.train_limit is not None and args.train_limit > 0:
            limit_n = min(args.train_limit, len(dataset))
            dataset.image_paths = dataset.image_paths[:limit_n]
            print(f"[Info] 简单文件夹模式：限制训练样本为 {limit_n} 张")
        if drop_last_train and len(dataset) < args.batch_size:
            drop_last_train = False
            print("[Warn] 训练集样本数小于批大小，已关闭drop_last以避免空批次。")
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=drop_last_train,
        )
        vis = gather_vis_samples(dataset, max_samples=min(4, len(dataset)))
        return train_loader, None, vis, dataset, None

    allowed_names = None
    image_labels: Optional[Dict[str, str]] = None
    if args.labels_csv is not None:
        image_labels = load_image_labels(
            args.labels_csv,
            image_column=args.csv_image_column,
            status_column=args.csv_status_column,
        )
        if not args.disable_label_filter:
            allowed_names = {name for name, label in image_labels.items() if label == str(args.csv_normal_label)}
            if not allowed_names:
                raise ValueError(f"No entries with label {args.csv_normal_label} found in {args.labels_csv}")

    train_mask_source = args.train_masks
    val_mask_source = args.val_masks

    if args.val_images is None:
        combined_imgs, combined_masks = resolve_image_mask_paths(
            args.train_images,
            train_mask_source,
            allowed_names=allowed_names,
        )
        (train_imgs, train_masks), (val_imgs, val_masks) = split_dataset_paths(
            combined_imgs,
            combined_masks,
            train_ratio=args.train_split,
            seed=args.split_seed,
            labels=image_labels,
        )
        print(f"Auto split dataset -> train: {len(train_imgs)}, val: {len(val_imgs)} "
              f"(ratio={args.train_split})")
    else:
        train_imgs, train_masks = resolve_image_mask_paths(
            args.train_images,
            train_mask_source,
            allowed_names=allowed_names,
        )
        val_imgs, val_masks = resolve_image_mask_paths(
            args.val_images,
            val_mask_source,
            allowed_names=allowed_names if not args.disable_label_filter else None,
        )
        print(f"Using explicit train/val datasets -> train: {len(train_imgs)}, val: {len(val_imgs)}")

    train_imgs, train_masks = limit_paths(train_imgs, train_masks, args.train_limit, "train")
    val_imgs, val_masks = limit_paths(val_imgs, val_masks, args.val_limit, "val")

    if len(train_imgs) == 0:
        raise RuntimeError("Training set is empty after applying filters/splits.")
    if len(val_imgs) == 0:
        raise RuntimeError("Validation set is empty after applying filters/splits.")

    train_dataset = RetinaDataset(
        image_paths=train_imgs,
        mask_paths=train_masks,
        train=True,
        grayscale=False,
        img_size=args.img_size,
        apply_color_jitter=args.retina_color_jitter,
    )
    val_dataset = RetinaDataset(
        image_paths=val_imgs,
        mask_paths=val_masks,
        train=False,
        grayscale=False,
        img_size=args.img_size,
        apply_color_jitter=False,
    )

    if drop_last_train and len(train_imgs) < args.batch_size:
        drop_last_train = False
        print("[Warn] 训练集样本数小于批大小，已关闭drop_last以避免空批次。")
    if drop_last_val and len(val_imgs) < args.batch_size:
        drop_last_val = False
        print("[Warn] 验证集样本数小于批大小，已关闭验证集drop_last。")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_optional_masks,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_optional_masks,
        drop_last=drop_last_val,
    )
    vis = gather_vis_samples(train_dataset, max_samples=min(4, len(train_dataset)))
    return train_loader, val_loader, vis, train_dataset, val_dataset


def _infer_upsample_mode(state_dict):
    has_bn_idx2 = any(k.startswith("decoder_core") and ".up.2." in k for k in state_dict.keys())
    has_conv_idx0 = any(k.startswith("decoder_core") and ".up.0.weight" in k for k in state_dict.keys())
    if has_bn_idx2:
        if has_conv_idx0:
            return "pixelshuffle"
        return "bilinear"
    return "deconv"


def load_ret_aae(weights_path: Path, device: torch.device) -> AAE:
    ckpt = torch.load(weights_path, map_location=device)
    if "model" not in ckpt:
        raise RuntimeError(f"Expected 'model' key in {weights_path}")
    cfg = ckpt.get("cfg", {})
    model = AAE(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        num_blocks=cfg.get("num_blocks", 5),
        base_channels=64,
        latent_dim=cfg.get("latent_dim", 256),
        img_size=224,
        upsample_mode=_infer_upsample_mode(ckpt["model"]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def adjust_inpaint_generator_channels(net: InpaintGenerator, color_mode: str, device: torch.device):
    if color_mode != "color":
        return net.to(device)
    first_conv = net.encoder[1]
    if first_conv.in_channels != 4:
        net.encoder[1] = nn.Conv2d(
            4, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
    last_conv = net.decoder[-1]
    if last_conv.out_channels != 3:
        net.decoder[-1] = nn.Conv2d(
            last_conv.in_channels, 3,
            kernel_size=last_conv.kernel_size,
            stride=last_conv.stride,
            padding=last_conv.padding,
            bias=last_conv.bias is not None,
        )
    return net.to(device)


def adjust_discriminator_channels(net: Discriminator, color_mode: str, device: torch.device):
    if color_mode != "color":
        return net.to(device)
    first_conv = net.conv[0]
    if first_conv.in_channels != 3:
        net.conv[0] = spectral_norm(nn.Conv2d(
            3, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        ))
    return net.to(device)


def build_aotgan_networks(color_mode: str, device: torch.device):
    netG = InpaintGenerator()
    netG = adjust_inpaint_generator_channels(netG, color_mode, device)
    netD = Discriminator()
    netD = adjust_discriminator_channels(netD, color_mode, device)
    return netG, netD


def train_aot_step(x_rgb: torch.Tensor,
                   x_gray: torch.Tensor,
                   mask: torch.Tensor,
                   netG: InpaintGenerator,
                   netD: Discriminator,
                   optimizer_G,
                   optimizer_D,
                   rec_loss_config,
                   rec_loss_dict,
                   adv_loss,
                   color_mode: str,
                   adv_weight: float = 0.01):
    if color_mode == "color":
        mask_rgb = mask.repeat(1, 3, 1, 1)
        transformed = (x_rgb * (1 - mask_rgb)) + mask_rgb
        pred = netG(transformed, mask)
        comp = (1 - mask_rgb) * x_rgb + mask_rgb * pred
        target = x_rgb
    else:
        mask_rgb = mask
        transformed = (x_gray * (1 - mask)) + mask
        pred = netG(transformed, mask)
        comp = (1 - mask) * x_gray + mask * pred
        target = x_gray

    rec_loss = sum(weight * rec_loss_dict[name](pred, target)
                   for name, weight in rec_loss_config.items())
    dis_loss, gen_loss = adv_loss(netD, comp, target, mask)

    optimizer_G.zero_grad()
    (rec_loss + adv_weight * gen_loss).backward(retain_graph=True)
    optimizer_G.step()

    optimizer_D.zero_grad()
    dis_loss.backward()
    optimizer_D.step()

    return rec_loss.item(), gen_loss.item(), dis_loss.item(), pred


@torch.no_grad()
def evaluate_rec_loss(loader,
                      netG: InpaintGenerator,
                      device: torch.device,
                      rec_loss_config,
                      rec_loss_dict,
                      color_mode: str):
    if loader is None:
        return None
    netG.eval()
    total = 0
    rec_total = 0.0
    for batch in loader:
        rgb, mask = split_batch(batch)
        rgb = rgb.to(device)
        mask = mask.to(device) if mask is not None else None
        if mask is None:
            raise RuntimeError("Validation masks missing; ensure precompute is enabled.")
        x_gray = TF.rgb_to_grayscale(rgb)
        if color_mode == "color":
            mask_rgb = mask.repeat(1, 3, 1, 1)
            transformed = (rgb * (1 - mask_rgb)) + mask_rgb
            pred = netG(transformed, mask)
            target = rgb
        else:
            transformed = (x_gray * (1 - mask)) + mask
            pred = netG(transformed, mask)
            target = x_gray
        rec_loss = sum(weight * rec_loss_dict[name](pred, target)
                       for name, weight in rec_loss_config.items())
        bsz = rgb.size(0)
        rec_total += rec_loss.item() * bsz
        total += bsz
    netG.train()
    return rec_total / max(total, 1)


def visualize_rgb(netG,
                  ret_aae,
                  samples_rgb: torch.Tensor,
                  sample_masks: Optional[torch.Tensor],
                  device: torch.device,
                  out_dir: Path,
                  epoch: int,
                  mask_threshold: float,
                  color_mode: str):
    netG.eval()
    ret_aae.eval()
    with torch.no_grad():
        rgb = samples_rgb.to(device)
        coarse_rgb, _ = ret_aae(rgb)
        x_gray = TF.rgb_to_grayscale(rgb)
        coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
        if sample_masks is not None:
            mask = sample_masks.to(device)
        else:
            mask = compute_mask_gpu(x_gray, coarse_gray, mask_threshold)
        if color_mode == "color":
            mask_rgb = mask.repeat(1, 3, 1, 1)
            transformed = (rgb * (1 - mask_rgb)) + mask_rgb
            pred = netG(transformed, mask)
            pred_rgb = pred.clamp(0.0, 1.0)
            mask_vis = mask_rgb
        else:
            transformed = (x_gray * (1 - mask)) + mask
            pred = netG(transformed, mask)

            rgb_gray = TF.rgb_to_grayscale(rgb)
            rgb_gray_exp = rgb_gray.expand_as(rgb)
            ratio = torch.where(rgb_gray_exp > 1e-6,
                                rgb / torch.clamp(rgb_gray_exp, min=1e-6),
                                torch.ones_like(rgb))
            pred_rgb = torch.clamp(pred.repeat(1, 3, 1, 1) * ratio, 0.0, 1.0)
            mask_vis = mask.repeat(1, 3, 1, 1)

        grid = torch.cat([rgb.cpu(), coarse_rgb.cpu(), pred_rgb.cpu(), mask_vis.cpu()], dim=0)
        grid = make_grid(grid, nrow=mask.shape[0], normalize=True, value_range=(0, 1))
        save_image(grid, out_dir / f"epoch_{epoch:04d}.png")
    netG.train()


def save_checkpoint(path: Path, netG, netD, optim_G, optim_D, epoch, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "netG": unwrap_module(netG).state_dict(),
        "netD": unwrap_module(netD).state_dict(),
        "optim_G": optim_G.state_dict(),
        "optim_D": optim_D.state_dict(),
        "stats": stats,
        "best_rec": stats.get("best_rec"),
    }, path)
    print(f"[Checkpoint] saved to {path}")


def save_training_log(log_rows: List[Dict], out_dir: Path):
    if not log_rows:
        print("No log rows to save.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train_log.csv"
    fieldnames = ["epoch", "train_rec", "train_gen", "train_dis", "val_rec", "lr_G", "lr_D"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)

    epochs = [row["epoch"] for row in log_rows]
    train_rec = [row["train_rec"] for row in log_rows]
    val_rec = [row.get("val_rec") for row in log_rows]
    has_val = any(v is not None for v in val_rec)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_rec, label="train rec loss")
    if has_val:
        val_curve = [v if v is not None else float("nan") for v in val_rec]
        plt.plot(epochs, val_curve, label="val rec loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AOT-GAN reconstruction loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    curve_path = out_dir / "loss_curve.png"
    plt.savefig(curve_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved training log to {csv_path}")
    print(f"Saved loss curve to {curve_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="AOT-GAN demo with RGB visualization + checkpoints")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Simple folder of images (legacy behavior).")
    parser.add_argument("--train-images", type=str, default=None,
                        help="Enable RetinaDataset loader (from AAE_S_1206) by pointing to image folder.")
    parser.add_argument("--train-masks", type=str, default=None,
                        help="Optional mask folder for RetinaDataset training split.")
    parser.add_argument("--val-images", type=str, default=None,
                        help="Validation image folder. If omitted, train-images will be split automatically.")
    parser.add_argument("--val-masks", type=str, default=None,
                        help="Optional mask folder for validation split.")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="CSV file for filtering/stratified split (same format as AAE_S_1206).")
    parser.add_argument("--csv-image-column", type=str, default="image_name")
    parser.add_argument("--csv-status-column", type=str, default="status_label")
    parser.add_argument("--csv-normal-label", type=str, default="0")
    parser.add_argument("--disable-label-filter", action="store_true")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train ratio when auto splitting RetinaDataset inputs.")
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--ret-weights", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--mask-threshold", type=float, default=0.153)
    parser.add_argument("--vis-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop if no metric improvement for N epochs; 0 disables early stopping.")
    parser.add_argument("--disable-parallel", action="store_true",
                        help="Force single-GPU mode even if multiple GPUs are visible.")
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "train_demo/aotgan_vis_rgb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-mode", type=str, choices=["gray", "color"], default="color",
                        help="Use grayscale (original behavior) or RGB training for AOT-GAN.")
    parser.add_argument("--retina-color-jitter", action="store_true",
                        help="Apply color jitter augmentation inside RetinaDataset.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training.")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="仅测试用：限制训练样本数量（按顺序取前N张）。")
    parser.add_argument("--val-limit", type=int, default=None,
                        help="仅测试用：限制验证样本数量（按顺序取前N张）。")
    parser.add_argument("--progress", action="store_true",
                        help="显示每轮训练批次进度条。")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="可选：掩码文件夹。未指定时会自动生成掩码到默认目录。")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    drop_last_train = num_gpus > 1 and not args.disable_parallel  # 避免多卡时末尾小批次被拆成空张量
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ret_aae = load_ret_aae(Path(args.ret_weights), device)

    if args.train_images is None:
        if args.image_dir is None:
            raise ValueError("Either --image-dir or --train-images must be provided")
        mask_dir = Path(args.mask_dir) if args.mask_dir is not None else out_dir / "auto_masks"
        print(f"[Mask] Using mask dir: {mask_dir}")
        base_dataset = ImageFolderDataset(Path(args.image_dir), img_size=args.img_size)
        limit_n = None
        if args.train_limit is not None and args.train_limit > 0:
            limit_n = min(args.train_limit, len(base_dataset))
            base_dataset.image_paths = base_dataset.image_paths[:limit_n]
            print(f"[Info] 简单文件夹模式：限制训练样本为 {limit_n} 张")
        generate_masks_to_dir(
            base_dataset,
            ret_aae,
            device,
            args.mask_threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mask_dir=mask_dir,
            progress=args.progress,
        )
        train_dataset = ImageFolderDataset(Path(args.image_dir), img_size=args.img_size, mask_dir=mask_dir)
        if limit_n is not None:
            train_dataset.image_paths = train_dataset.image_paths[:limit_n]
            if train_dataset.mask_paths is not None:
                train_dataset.mask_paths = train_dataset.mask_paths[:limit_n]
        if drop_last_train and len(train_dataset) < args.batch_size:
            drop_last_train = False
            print("[Warn] 训练集样本数小于批大小，已关闭drop_last以避免空批次。")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=drop_last_train,
            collate_fn=collate_with_optional_masks,
        )
        val_loader = None
        val_dataset = None
    else:
        train_loader, val_loader, vis_samples, train_dataset, val_dataset = prepare_dataloaders(
            args,
            drop_last_train=drop_last_train,
            drop_last_val=drop_last_train,
        )
        precompute_and_cache_masks(
            train_dataset,
            ret_aae,
            device,
            args.mask_threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            progress=args.progress,
        )
        precompute_and_cache_masks(
            val_dataset,
            ret_aae,
            device,
            args.mask_threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            progress=args.progress,
        )
    ret_aae.eval()
    # 可视化样本/掩码来自缓存后的版本，避免与训练数据不一致
    vis_samples = gather_vis_samples(train_dataset, max_samples=min(4, len(train_dataset)))
    vis_masks = None
    if train_dataset is not None and hasattr(train_dataset, "cached_masks") and train_dataset.cached_masks is not None:
        vis_count = vis_samples.size(0)
        vis_masks = train_dataset.cached_masks[:vis_count]
    netG, netD = build_aotgan_networks(args.color_mode, device)
    multi_gpu = (device.type == "cuda" and num_gpus > 1 and args.batch_size >= num_gpus
                 and not args.disable_parallel)
    if args.disable_parallel and num_gpus > 1:
        print(f"[Info] 检测到 {num_gpus} 张GPU，但已通过 --disable-parallel 强制单卡。")
    if device.type == "cuda" and num_gpus > 1 and args.batch_size < num_gpus:
        print(f"[Warn] 批大小({args.batch_size})小于GPU数({num_gpus})，已禁用DataParallel以避免空分片错误。")
    if multi_gpu:
        print(f"Using {num_gpus} GPUs via DataParallel")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    optim_G = torch.optim.Adam(netG.parameters(), lr=5e-5, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))
    rec_loss_config = {"L1": 1.0, "Style": 250.0, "Perceptual": 0.1}
    rec_loss_dict = {name: getattr(loss_module, name)() for name in rec_loss_config}
    adv_loss = loss_module.smgan()

    best_metric = float("inf")
    no_improve_epochs = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint '{resume_path}' not found")
        ckpt = torch.load(resume_path, map_location=device)
        if "netG" not in ckpt or "netD" not in ckpt:
            raise RuntimeError(f"Checkpoint {resume_path} missing generator/discriminator weights")
        unwrap_module(netG).load_state_dict(ckpt["netG"])
        unwrap_module(netD).load_state_dict(ckpt["netD"])
        if "optim_G" in ckpt and "optim_D" in ckpt:
            optim_G.load_state_dict(ckpt["optim_G"])
            optim_D.load_state_dict(ckpt["optim_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_rec", ckpt.get("stats", {}).get("rec", float("inf")))
        print(f"[Resume] Loaded checkpoint '{resume_path}' -> start_epoch={start_epoch}, best_rec={best_metric:.4f}")

    log_rows: List[Dict] = []
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if start_epoch > args.epochs:
        print(f"Start epoch ({start_epoch}) exceeds target epochs ({args.epochs}). Skipping training.")

    for epoch in range(start_epoch, args.epochs + 1):
        netG.train()
        epoch_stats = {"rec": 0.0, "gen": 0.0, "dis": 0.0, "n": 0}

        train_iter = tqdm(train_loader, desc=f"Train {epoch}", leave=False) if args.progress else train_loader
        for batch in train_iter:
            rgb, mask = split_batch(batch)
            rgb = rgb.to(device)
            mask = mask.to(device) if mask is not None else None
            if mask is None:
                with torch.no_grad():
                    coarse_rgb, _ = ret_aae(rgb)
                x_gray = TF.rgb_to_grayscale(rgb)
                coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
                mask = compute_mask_gpu(x_gray, coarse_gray, args.mask_threshold)
            x_gray = TF.rgb_to_grayscale(rgb)

            rec_l, gen_l, dis_l, _ = train_aot_step(
                rgb, x_gray, mask, netG, netD, optim_G, optim_D,
                rec_loss_config, rec_loss_dict, adv_loss, args.color_mode)
            bsz = x_gray.size(0)
            epoch_stats["rec"] += rec_l * bsz
            epoch_stats["gen"] += gen_l * bsz
            epoch_stats["dis"] += dis_l * bsz
            epoch_stats["n"] += bsz
            if args.progress and epoch_stats["n"] > 0:
                cur_rec = epoch_stats["rec"] / epoch_stats["n"]
                cur_gen = epoch_stats["gen"] / epoch_stats["n"]
                cur_dis = epoch_stats["dis"] / epoch_stats["n"]
                train_iter.set_postfix(rec=f"{cur_rec:.4f}", gen=f"{cur_gen:.4f}", dis=f"{cur_dis:.4f}")

        n = max(epoch_stats["n"], 1)
        avg_rec = epoch_stats["rec"] / n
        avg_gen = epoch_stats["gen"] / n
        avg_dis = epoch_stats["dis"] / n
        val_rec = evaluate_rec_loss(
            val_loader, netG, device,
            rec_loss_config, rec_loss_dict, args.color_mode)
        msg = f"[Epoch {epoch}/{args.epochs}] rec={avg_rec:.4f}, gen={avg_gen:.4f}, dis={avg_dis:.4f}"
        if val_rec is not None:
            msg += f", val_rec={val_rec:.4f}"
        print(msg)

        metric = val_rec if val_rec is not None else avg_rec
        improved = metric < best_metric
        if improved:
            best_metric = metric
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        stats = {
            "rec": avg_rec,
            "gen": avg_gen,
            "dis": avg_dis,
            "color_mode": args.color_mode,
            "val_rec": val_rec,
            "best_rec": best_metric,
        }

        save_checkpoint(ckpt_dir / "latest.pth", netG, netD, optim_G, optim_D, epoch, stats)
        if improved:
            save_checkpoint(ckpt_dir / "best.pth", netG, netD, optim_G, optim_D, epoch, stats)

        log_rows.append({
            "epoch": epoch,
            "train_rec": avg_rec,
            "train_gen": avg_gen,
            "train_dis": avg_dis,
            "val_rec": val_rec,
            "lr_G": optim_G.param_groups[0]["lr"],
            "lr_D": optim_D.param_groups[0]["lr"],
        })

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"[Early Stop] No improvement for {no_improve_epochs} epochs "
                  f"(patience={args.early_stop_patience}). Stopping training.")
            break

        if epoch % args.vis_interval == 0:
            visualize_rgb(netG, ret_aae, vis_samples, vis_masks,
                          device, out_dir, epoch, args.mask_threshold, args.color_mode)

    save_training_log(log_rows, out_dir)


if __name__ == "__main__":
    main()
