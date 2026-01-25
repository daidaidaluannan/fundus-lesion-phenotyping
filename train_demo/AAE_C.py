#!/usr/bin/env python
# aae_retina.py
# PyTorch 2.x, torchvision >= 0.15
import os
import math
import random
import csv
import argparse
from typing import Optional, Tuple, List, Dict, Sequence, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import torchvision.models as models
#from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.utils import make_grid, save_image

# 学习率调度器
try:
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
except ImportError:
    SequentialLR = LinearLR = ConstantLR = None
from torch.optim.lr_scheduler import LambdaLR

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
from torch.utils.data import Subset


# ----------------------------
# Model building blocks
# ----------------------------

class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(2 * mid, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        y = self.mix(y)
        y = y + self.skip(x)
        return self.act(y)


class EfficientMultiScaleAttention(nn.Module):
    def __init__(self, channels: int, ks: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False) for k in ks
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)
        att = 0
        for conv in self.convs:
            att = att + conv(y)
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msrb = MultiScaleResidualBlock(out_ch, out_ch)
        self.ema = EfficientMultiScaleAttention(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.msrb(x)
        x = self.ema(x)
        return x


class DecoderBlock(nn.Module):
    """
    支持三种上采样方式：
      - deconv: ConvTranspose2d (原始实现)
      - bilinear: Upsample(bilinear) + Conv3x3
      - pixelshuffle: Conv3x3(out_ch*4) + PixelShuffle(2)
    """
    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True, mode: str = "deconv"):
        super().__init__()
        assert mode in ("deconv", "bilinear", "pixelshuffle")
        self.msrb = MultiScaleResidualBlock(in_ch, in_ch)
        self.ema = EfficientMultiScaleAttention(in_ch)
        self.mode = mode
        if upsample:
            if mode == "deconv":
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            elif mode == "bilinear":
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:  # pixelshuffle
                self.up = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        x = self.msrb(x)
        x = self.ema(x)
        x = self.up(x)
        return x


class WeightScheduler:
    def __init__(self,
                 loss_module,
                 vessel_start=8.0, vessel_end=20.0,
                 edge_start=0.0, edge_end=1.0,
                 total_epochs=100,
                 warmup_epochs=0,
                 mode="linear"):
        self.loss_module = loss_module
        self.vs0, self.vs1 = float(vessel_start), float(vessel_end)
        self.es0, self.es1 = float(edge_start), float(edge_end)
        self.T = max(1, int(total_epochs))
        self.warm = max(0, int(warmup_epochs))
        assert mode in ("linear", "cosine")
        self.mode = mode

    def _interp(self, t):
        if self.mode == "linear":
            return t
        return 0.5 * (1 - math.cos(math.pi * t))

    def values_at(self, epoch):
        if epoch <= self.warm:
            return self.vs0, self.es0
        num = min(max(epoch - self.warm, 0), max(self.T - self.warm, 1))
        den = max(self.T - self.warm, 1)
        t = self._interp(num / den)
        v = self.vs0 + (self.vs1 - self.vs0) * t
        e = self.es0 + (self.es1 - self.es0) * t
        return v, e

    def step(self, epoch, verbose=False):
        v, e = self.values_at(epoch)
        if hasattr(self.loss_module, "vessel_weight"):
            self.loss_module.vessel_weight = float(v)
        if hasattr(self.loss_module, "edge_extra_weight"):
            self.loss_module.edge_extra_weight = float(e)
        if verbose:
            print(f"[WeightScheduler] epoch={epoch} vessel_weight={v:.3f} edge_extra_weight={e:.3f}")


class AAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_blocks: int = 5,
                 base_channels: int = 64,
                 latent_dim: int = 256,
                 img_size: int = 224,
                 upsample_mode: str = "deconv"):
        super().__init__()
        assert img_size % (2 ** num_blocks) == 0, "img_size must be divisible by 2**num_blocks"
        assert upsample_mode in ("deconv", "bilinear", "pixelshuffle")
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_blocks = num_blocks
        self.upsample_mode = upsample_mode

        chs = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8][:num_blocks]
        enc = []
        in_ch = in_channels
        for c in chs:
            enc.append(EncoderBlock(in_ch, c, downsample=True))
            in_ch = c
        self.encoder = nn.Sequential(*enc)

        self.spatial = img_size // (2 ** num_blocks)
        self.enc_out_ch = chs[-1]
        enc_flat = self.enc_out_ch * self.spatial * self.spatial

        self.to_latent = nn.Linear(enc_flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, enc_flat)

        dec_chs = chs[::-1]
        dec = []
        for i in range(num_blocks):
            in_c = dec_chs[i]
            out_c = dec_chs[i+1] if i+1 < len(dec_chs) else base_channels
            dec.append(DecoderBlock(in_c, out_c, upsample=True, mode=upsample_mode))
        self.decoder_core = nn.Sequential(*dec)

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(1)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(z.size(0), self.enc_out_ch, self.spatial, self.spatial)
        h = self.decoder_core(h)
        xrec = self.final(h)
        return xrec

    def forward(self, x):
        z = self.encode(x)
        xrec = self.decode(z)
        return xrec, z


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear( hidden, 1 ),
        )

    def forward(self, z):
        return self.net(z).view(-1)


# ----------------------------
# Losses
# ----------------------------


def build_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_factor):
    """Create a LR scheduler with linear warmup if supported, otherwise fallback."""
    warmup_epochs = int(max(0, warmup_epochs))
    total_epochs = int(max(1, total_epochs))
    min_factor = float(min_factor)

    if SequentialLR is not None and LinearLR is not None and ConstantLR is not None:
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=min_factor, total_iters=warmup_epochs),
                ConstantLR(optimizer, factor=1.0, total_iters=max(1, total_epochs - warmup_epochs)),
            ],
            milestones=[warmup_epochs]
        )

    if warmup_epochs == 0:
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    def lr_lambda(epoch):
        if epoch >= warmup_epochs:
            return 1.0
        return min_factor + (1.0 - min_factor) * (epoch / max(1, warmup_epochs))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

class WeightedL1Loss(nn.Module):
    def __init__(
        self,
        vessel_weight: float = 12.29,
        edge_extra_weight: float = 0.0,
        edge_band_width: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.vessel_weight = float(vessel_weight)
        self.edge_extra_weight = float(edge_extra_weight)
        self.edge_band_width = int(max(0, edge_band_width))
        self.reduction = reduction
        k = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        self.register_buffer("morph_kernel", k)

    @torch.no_grad()
    def _edge_ring(self, mask: torch.Tensor) -> torch.Tensor:
        if self.edge_band_width <= 0:
            return torch.zeros_like(mask)

        k = self.morph_kernel.to(mask)
        def dilate(x):
            y = x
            for _ in range(self.edge_band_width):
                s = F.conv2d(y, k, padding=1, groups=1)
                y = (s > 0).float()
            return y

        def erode(x):
            y = x
            for _ in range(self.edge_band_width):
                s = F.conv2d(y, k, padding=1, groups=1)
                y = (s >= 9.0).float()
            return y

        dil = dilate(mask)
        ero = erode(mask)
        edge = (dil - ero).clamp_(0, 1)
        return edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            loss = torch.abs(pred - target)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).to(pred.dtype)

        w = torch.ones_like(pred)
        w = torch.where(mask > 0.5, torch.full_like(w, self.vessel_weight), w)

        if self.edge_extra_weight > 0 and self.edge_band_width > 0:
            with torch.no_grad():
                edge = self._edge_ring(mask)
            edge = edge.expand_as(pred)
            w = torch.where(edge > 0.5, w * (1.0 + self.edge_extra_weight), w)

        abs_err = torch.abs(pred - target)
        if self.reduction == "mean":
            return (abs_err * w).sum() / w.sum().clamp_min(1.0)
        elif self.reduction == "sum":
            return (abs_err * w).sum()
        else:
            return abs_err * w


class VGGPerceptual(nn.Module):
    def __init__(self, layer_index: int = 22, requires_grad: bool = False, device: str = "cpu"):
        super().__init__()
        device = torch.device(device)
        #vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg = models.vgg16_bn(pretrained=True).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(layer_index + 1)])
        if not requires_grad:
            for p in self.slice.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))
        self.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.size(1) == 1:
            pred = pred.repeat(1,3,1,1)
            target = target.repeat(1,3,1,1)
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        f_pred = self.slice(pred_n)
        f_tgt = self.slice(target_n)
        return F.mse_loss(f_pred, f_tgt)


def unwrap_module(module: nn.Module) -> nn.Module:
    """Return the underlying module when wrapped by DataParallel."""
    return module.module if isinstance(module, nn.DataParallel) else module


# ----------------------------
# Data
# ----------------------------

class RetinaDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: Optional[List[str]] = None,
                 train: bool = True,
                 grayscale: bool = False,
                 img_size: int = 224,
                 apply_color_jitter: bool = False):
        assert mask_paths is None or len(mask_paths) == len(image_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.train = train
        self.grayscale = grayscale
        self.img_size = img_size
        self.apply_color_jitter = apply_color_jitter
        self.cj = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if apply_color_jitter else None

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L" if self.grayscale else "RGB")
        m = None
        if self.mask_paths is not None:
            m = Image.open(self.mask_paths[idx]).convert("L")
        return img, m

    def __getitem__(self, idx):
        img, m = self._load_pair(idx)
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        if m is not None:
            m = TF.resize(m, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                if m is not None:
                    m = TF.hflip(m)
            angle = random.uniform(0.0, 15.0)
            if random.random() < 0.5:
                angle = -angle
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
        return img_t, mask_t


def collate_with_optional_masks(batch):
    imgs, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    if all(m is None for m in masks):
        return imgs, None
    else:
        masks = torch.stack([m for m in masks], dim=0)
        return imgs, masks


def load_allowed_image_names(csv_path: str,
                             image_column: str = "image_name",
                             status_column: str = "status_label",
                             normal_label: str = "0") -> Set[str]:
    allowed = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or image_column not in reader.fieldnames or status_column not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must contain columns '{image_column}' and '{status_column}'")
        for row in reader:
            if row.get(status_column) is None:
                continue
            if str(row[status_column]).strip() == str(normal_label):
                name = row.get(image_column)
                if name:
                    allowed.add(name.strip())
    return allowed


def split_dataset_paths(image_paths: List[str],
                        mask_paths: Optional[List[str]],
                        train_ratio: float,
                        seed: int) -> Tuple[Tuple[List[str], Optional[List[str]]], Tuple[List[str], Optional[List[str]]]]:
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

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_idx = max(1, min(total - 1, int(round(total * ratio))))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    def gather(idxs):
        imgs = [image_paths[i] for i in idxs]
        masks = [mask_paths[i] for i in idxs] if mask_paths is not None else None
        return imgs, masks

    return gather(train_idx), gather(val_idx)


def resolve_image_mask_paths(image_source: Union[str, Sequence[str]],
                             mask_source: Optional[Union[str, Sequence[str]]],
                             allowed_names: Optional[Set[str]] = None) -> Tuple[List[str], Optional[List[str]]]:
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


# ----------------------------
# Training utilities
# ----------------------------

@torch.no_grad()
def ssim_torch(x, y, C1=0.01**2, C2=0.03**2):
    def gaussian_window(kernel_size=11, sigma=1.5, channels=1):
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords**2) / (2*sigma*sigma))
        g = (g / g.sum()).unsqueeze(1)
        w = g @ g.t()
        w = w / w.sum()
        w = w.view(1,1,kernel_size,kernel_size).repeat(channels,1,1,1)
        return w

    B, C, H, W = x.shape
    w = gaussian_window(channels=C).to(x.device)
    mu_x = F.conv2d(x, w, padding=5, groups=C)
    mu_y = F.conv2d(y, w, padding=5, groups=C)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x*x, w, padding=5, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y*y, w, padding=5, groups=C) - mu_y2
    sigma_xy = F.conv2d(x*y, w, padding=5, groups=C) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


def adversarial_losses(d_out_real: Optional[torch.Tensor] = None,
                       d_out_fake: Optional[torch.Tensor] = None,
                       label_smooth_real: float = 0.9):
    bce = nn.BCEWithLogitsLoss()
    d_loss = None
    g_loss = None
    if d_out_real is not None and d_out_fake is not None:
        real_t = torch.full_like(d_out_real, fill_value=label_smooth_real)
        fake_t = torch.zeros_like(d_out_fake)
        d_loss = bce(d_out_real, real_t) + bce(d_out_fake, fake_t)
    if d_out_fake is not None:
        g_loss = bce(d_out_fake, torch.ones_like(d_out_fake))
    return d_loss, g_loss


def build_models_and_opts(modality: str, device: torch.device, upsample_mode: str,
                          ae_lr: float, d_lr: float, weight_decay: float,
                          lambda_adv: float, lambda_perc: float,
                          latent_dim: int):
    is_cfp = modality.upper() == "CFP"
    cfg = {}
    if is_cfp:
        cfg.update(dict(
            in_channels=3, out_channels=3, num_blocks=5,
            lr_ae=ae_lr, lr_d=d_lr, weight_decay=weight_decay,
            lambda_adv=lambda_adv, lambda_perc=lambda_perc, vessel_weight=12.29,
            latent_dim=latent_dim
        ))
    else:
        cfg.update(dict(
            in_channels=1, out_channels=1, num_blocks=4,
            lr_ae=ae_lr, lr_d=d_lr, weight_decay=1e-3,
            lambda_adv=lambda_adv, lambda_perc=0.0 if lambda_perc is None else lambda_perc, vessel_weight=1.0,
            latent_dim=latent_dim
        ))

    model = AAE(in_channels=cfg["in_channels"],
                out_channels=cfg["out_channels"],
                num_blocks=cfg["num_blocks"],
                base_channels=64,
                latent_dim=cfg["latent_dim"],
                img_size=224,
                upsample_mode=upsample_mode).to(device)
    disc = LatentDiscriminator(latent_dim=cfg["latent_dim"], hidden=512).to(device)

    optim_ae = torch.optim.AdamW(model.parameters(), lr=cfg["lr_ae"], weight_decay=cfg["weight_decay"])
    optim_d = torch.optim.AdamW(disc.parameters(), lr=cfg["lr_d"], weight_decay=cfg["weight_decay"])
    return model, disc, optim_ae, optim_d, cfg


def train_one_epoch(model: AAE,
                    disc: LatentDiscriminator,
                    loader: DataLoader,
                    optim_ae,
                    optim_d,
                    device: torch.device,
                    cfg: Dict,
                    perceptual: Optional[VGGPerceptual] = None,
                    modality: str = "CFP",
                    log_interval: int = 50,
                    loss_l1: Optional[nn.Module] = None):
    model.train()
    disc.train()
    is_cfp = modality.upper() == "CFP"
    w_l1 = loss_l1
    avg = {"loss": 0.0, "l1": 0.0, "perc": 0.0, "adv_g": 0.0, "adv_d": 0.0, "ssim": 0.0}
    n = 0

    for it, batch in enumerate(loader):
        imgs, masks = batch
        imgs = imgs.to(device)
        masks = masks.to(device) if (is_cfp and masks is not None) else None

        # D step
        with torch.no_grad():
            _, z_fake = model(imgs)
        z_real = torch.randn_like(z_fake)
        d_real = disc(z_real)
        d_fake = disc(z_fake.detach())
        d_loss, _ = adversarial_losses(d_out_real=d_real, d_out_fake=d_fake, label_smooth_real=0.9)

        optim_d.zero_grad(set_to_none=True)
        d_loss.backward()
        optim_d.step()

        # AE step
        xrec, z_fake = model(imgs)
        d_fake = disc(z_fake)
        _, g_adv = adversarial_losses(d_out_fake=d_fake, label_smooth_real=0.9)

        if w_l1 is not None:
            l1 = w_l1(xrec, imgs, mask=masks if is_cfp else None)
        else:
            l1 = F.l1_loss(xrec, imgs, reduction="mean")

        perc = perceptual(xrec, imgs) if (is_cfp and perceptual is not None) else torch.zeros(1, device=device)
        if isinstance(perc, torch.Tensor):
            perc = perc.mean()
        loss = l1 + cfg["lambda_perc"] * perc + cfg["lambda_adv"] * g_adv

        optim_ae.zero_grad(set_to_none=True)
        loss.backward()
        optim_ae.step()

        with torch.no_grad():
            ssim = ssim_torch(xrec, imgs).item()

        bsz = imgs.size(0)
        avg["loss"] += loss.item() * bsz
        avg["l1"] += l1.item() * bsz
        avg["perc"] += float(perc.item()) * bsz
        avg["adv_g"] += g_adv.item() * bsz
        avg["adv_d"] += d_loss.item() * bsz
        avg["ssim"] += ssim * bsz
        n += bsz

        if (it + 1) % log_interval == 0:
            l1_val = l1.item() if isinstance(l1, torch.Tensor) else float(l1)
            print(f"[iter {it+1}] loss={loss.item():.4f} | l1={l1_val:.4f} | perc={float(perc.item()):.4f} | "
                  f"adv_g={g_adv.item():.4f} | adv_d={d_loss.item():.4f} | ssim={ssim:.4f}")

    for k in avg:
        avg[k] /= max(n, 1)
    return avg


@torch.no_grad()
def evaluate(model: AAE, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    ssim_sum = 0.0
    l1_sum = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        xrec, _ = model(imgs)
        ssim_sum += ssim_torch(xrec, imgs).item() * imgs.size(0)
        l1_sum += F.l1_loss(xrec, imgs, reduction="mean").item() * imgs.size(0)
        total += imgs.size(0)
    return {"ssim": ssim_sum / total, "l1": l1_sum / total}


# ----------------------------
# Visualization & logging utils
# ----------------------------

@torch.no_grad()
def save_recon_grid(model: AAE,
                    images: torch.Tensor,
                    device: torch.device,
                    out_dir: str,
                    epoch: int,
                    num_images: int = 8) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    n = min(num_images, images.size(0))
    imgs = images[:n].to(device)
    recons, _ = model(imgs)
    grid_in = make_grid(imgs, nrow=n, padding=2)
    grid_out = make_grid(recons, nrow=n, padding=2)
    comp = torch.cat([grid_in, grid_out], dim=1)
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    save_image(comp, save_path)
    return save_path


def save_logs_and_curves(log_rows: List[Dict], out_dir: str, modality: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"train_log_{modality.lower()}.csv")
    fieldnames = ["epoch", "train_loss", "train_l1", "train_perc", "train_adv_g", "train_adv_d",
                  "train_ssim", "val_ssim", "val_l1", "lr_ae", "lr_d"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)

    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    train_l1 = [r["train_l1"] for r in log_rows]
    val_l1 = [r["val_l1"] for r in log_rows]
    train_ssim = [r["train_ssim"] for r in log_rows]
    val_ssim = [r["val_ssim"] for r in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, train_l1, label="train L1")
    plt.plot(epochs, val_l1, label="val L1")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss curves ({modality})")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    loss_curve_path = os.path.join(out_dir, f"loss_curve_{modality.lower()}.png")
    plt.savefig(loss_curve_path, bbox_inches="tight", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_ssim, label="train SSIM")
    plt.plot(epochs, val_ssim, label="val SSIM")
    plt.xlabel("Epoch"); plt.ylabel("SSIM"); plt.title(f"SSIM curves ({modality})")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    ssim_curve_path = os.path.join(out_dir, f"ssim_curve_{modality.lower()}.png")
    plt.savefig(ssim_curve_path, bbox_inches="tight", dpi=150); plt.close()

    print(f"Saved training log to {csv_path}")
    print(f"Saved loss curve to {loss_curve_path}")
    print(f"Saved SSIM curve to {ssim_curve_path}")


def make_dataloader(image_source: Union[str, Sequence[str]],
                    mask_source: Optional[Union[str, Sequence[str]]],
                    modality: str,
                    train: bool,
                    batch_size: int = 16,
                    num_workers: int = 6,
                    img_size: int = 224,
                    subset_ratio: Optional[float] = None,
                    subset_count: Optional[int] = None,
                    subset_seed: int = 42,
                    allowed_names: Optional[Set[str]] = None) -> DataLoader:
    img_paths, mask_paths = resolve_image_mask_paths(image_source, mask_source, allowed_names=allowed_names)

    is_cfp = modality.upper() == "CFP"
    ds = RetinaDataset(
        image_paths=img_paths,
        mask_paths=mask_paths,
        train=train,
        grayscale=(modality.upper() == "OCT"),
        img_size=img_size,
        apply_color_jitter=is_cfp and train
    )

    if subset_ratio is not None or subset_count is not None:
        import math as _math, random as _random
        n = len(ds)
        k = subset_count if subset_count is not None else max(1, int(_math.ceil(n * float(subset_ratio))))
        rng = _random.Random(subset_seed)
        indices = list(range(n))
        rng.shuffle(indices)
        indices = indices[:k]
        ds = Subset(ds, indices)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_with_optional_masks
    )


def save_checkpoint(path: str,
                    model: AAE,
                    disc: LatentDiscriminator,
                    optim_ae,
                    optim_d,
                    epoch: int,
                    cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = unwrap_module(model)
    disc_to_save = unwrap_module(disc)
    torch.save({
        "model": model_to_save.state_dict(),
        "disc": disc_to_save.state_dict(),
        "optim_ae": optim_ae.state_dict(),
        "optim_d": optim_d.state_dict(),
        "epoch": epoch,
        "cfg": cfg
    }, path)
    print(f"Saved checkpoint to {path}")


# ----------------------------
# CLI and main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AAE Retina Training / Resume / Finetune")
    p.add_argument("--modality", type=str, default="CFP", choices=["CFP", "OCT"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--img-size", type=int, default=224)

    p.add_argument("--train-images", type=str, required=True)
    p.add_argument("--train-masks", type=str, default=None)
    p.add_argument("--val-images", type=str, default=None,
                   help="Optional validation image dir; if omitted, split from train-images")
    p.add_argument("--val-masks", type=str, default=None)

    p.add_argument("--out-dir", type=str, default="./checkpoints_0831")
    p.add_argument("--upsample-mode", type=str, default="deconv", choices=["deconv", "bilinear", "pixelshuffle"])
    p.add_argument("--latent-dim", type=int, default=256,
                   help="潜在空间维度，需与判别器保持一致")

    # Filtering & splitting
    p.add_argument("--labels-csv", type=str, default=None, help="CSV file for filtering normal images")
    p.add_argument("--csv-image-column", type=str, default="image_name")
    p.add_argument("--csv-status-column", type=str, default="status_label")
    p.add_argument("--csv-normal-label", type=str, default="0")
    p.add_argument("--train-split", type=float, default=0.8,
                   help="Train ratio when auto-splitting a single folder")
    p.add_argument("--split-seed", type=int, default=2024,
                   help="Seed for deterministic train/val split")

    # Learning rates and losses
    p.add_argument("--ae-lr", type=float, default=None, help="AE learning rate; default depends on modality")
    p.add_argument("--d-lr", type=float, default=None, help="D learning rate; default depends on modality")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lambda-adv", type=float, default=1e-3)
    p.add_argument("--lambda-perc", type=float, default=0.2)

    # LR warmup
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--min-lr-factor", type=float, default=1e-3)

    # Weighted L1 scheduler
    p.add_argument("--edge-band", type=int, default=1)
    p.add_argument("--vessel-start", type=float, default=8.0)
    p.add_argument("--vessel-end", type=float, default=20.0)
    p.add_argument("--edge-start", type=float, default=0.0)
    p.add_argument("--edge-end", type=float, default=1.0)
    p.add_argument("--weight-warmup-epochs", type=int, default=10)
    p.add_argument("--weight-sched-mode", type=str, default="linear", choices=["linear", "cosine"])

    # Resume / finetune
    p.add_argument("--resume", type=str, default=None, help="resume path: load model+disc+opt+sched and continue")
    p.add_argument("--finetune", type=str, default=None, help="finetune path: load model weights only (strict=False)")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--subset-ratio", type=float, default=None)
    p.add_argument("--subset-count", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    modality = args.modality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default LRs if not provided
    if args.ae_lr is None:
        args.ae_lr = 5e-4 if modality.upper() == "CFP" else 1e-4
    if args.d_lr is None:
        args.d_lr = 1e-4 if modality.upper() == "CFP" else 5e-5

    allowed_names = None
    if args.labels_csv is not None:
        allowed_names = load_allowed_image_names(
            args.labels_csv,
            image_column=args.csv_image_column,
            status_column=args.csv_status_column,
            normal_label=args.csv_normal_label,
        )
        if not allowed_names:
            raise ValueError(f"No entries with label {args.csv_normal_label} found in {args.labels_csv}")

    train_mask_source = args.train_masks if modality == "CFP" else None
    val_mask_source = args.val_masks if modality == "CFP" else None

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
        )
        print(f"Auto split dataset -> train: {len(train_imgs)}, val: {len(val_imgs)} (ratio={args.train_split})")
        train_loader = make_dataloader(train_imgs, train_masks,
                                       modality, train=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, img_size=args.img_size,
                                       subset_ratio=args.subset_ratio, subset_count=args.subset_count,
                                       subset_seed=args.subset_seed)
        val_loader = make_dataloader(val_imgs, val_masks,
                                     modality, train=False, batch_size=args.batch_size,
                                     num_workers=args.num_workers, img_size=args.img_size)
    else:
        train_loader = make_dataloader(args.train_images, train_mask_source,
                                       modality, train=True, batch_size=args.batch_size,
                                       num_workers=args.num_workers, img_size=args.img_size,
                                       subset_ratio=args.subset_ratio, subset_count=args.subset_count,
                                       subset_seed=args.subset_seed, allowed_names=allowed_names)
        val_loader = make_dataloader(args.val_images, val_mask_source,
                                     modality, train=False, batch_size=args.batch_size,
                                     num_workers=args.num_workers, img_size=args.img_size,
                                     subset_ratio=0.2 if args.subset_ratio is None and args.subset_count is None else None,
                                     allowed_names=allowed_names)

    # Build models and opts
    model, disc, optim_ae, optim_d, cfg = build_models_and_opts(
        modality, device, args.upsample_mode, args.ae_lr, args.d_lr, args.weight_decay,
        args.lambda_adv, args.lambda_perc, args.latent_dim
    )

    # Perceptual loss (CFP only)
    perceptual = VGGPerceptual(layer_index=22, requires_grad=False, device=device) if modality.upper() == "CFP" else None

    multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1
    if multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)

    # Prepare fixed vis batch
    try:
        vis_imgs, _ = next(iter(val_loader))
    except StopIteration:
        vis_imgs, _ = next(iter(train_loader))
    vis_imgs = vis_imgs.cpu()

    # Output dirs
    base_dir = args.out_dir
    vis_dir = os.path.join(base_dir, f"vis_{modality.lower()}")
    os.makedirs(vis_dir, exist_ok=True)

    # Weighted L1 and scheduler
    cfg.setdefault("edge_band_width", args.edge_band)
    cfg.setdefault("vessel_start", args.vessel_start)
    cfg.setdefault("vessel_end", args.vessel_end)
    cfg.setdefault("edge_start", args.edge_start)
    cfg.setdefault("edge_end", args.edge_end)
    cfg.setdefault("weight_warmup_epochs", args.weight_warmup_epochs)
    cfg.setdefault("weight_sched_mode", args.weight_sched_mode)

    w_l1 = WeightedL1Loss(
        vessel_weight=cfg["vessel_start"],
        edge_extra_weight=cfg["edge_start"],
        edge_band_width=cfg["edge_band_width"],
        reduction="mean",
    ).to(device)

    weight_sched = WeightScheduler(
        loss_module=w_l1,
        vessel_start=cfg["vessel_start"], vessel_end=cfg["vessel_end"],
        edge_start=cfg["edge_start"], edge_end=cfg["edge_end"],
        total_epochs=args.epochs,
        warmup_epochs=cfg["weight_warmup_epochs"],
        mode=cfg["weight_sched_mode"],
    )

    # LR warmup schedulers
    warmup_epochs = int(min(args.warmup_epochs, args.epochs))
    min_factor = args.min_lr_factor
    sched_ae = build_warmup_scheduler(optim_ae, warmup_epochs, args.epochs, min_factor)
    sched_d = build_warmup_scheduler(optim_d, warmup_epochs, args.epochs, min_factor)

    start_epoch = 1
    resume_mode = False
    finetune_mode = False

    # Resume
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        unwrap_module(model).load_state_dict(ckpt["model"])
        unwrap_module(disc).load_state_dict(ckpt["disc"])
        optim_ae.load_state_dict(ckpt["optim_ae"])
        optim_d.load_state_dict(ckpt["optim_d"])
        start_epoch = ckpt.get("epoch", 0) + 1
        resume_mode = True
        print(f"[Resume] Loaded from {args.resume}, start_epoch={start_epoch}")

    # Finetune
    if args.finetune is not None and os.path.isfile(args.finetune):
        ckpt = torch.load(args.finetune, map_location=device)
        missing, unexpected = unwrap_module(model).load_state_dict(ckpt["model"], strict=False)
        print(f"[Finetune] Loaded weights from {args.finetune}")
        if missing or unexpected:
            print("  Missing keys:", missing)
            print("  Unexpected keys:", unexpected)
        # 重建优化器和调度器（避免继承旧动量）
        optim_ae = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        optim_d = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, weight_decay=args.weight_decay)
        sched_ae = build_warmup_scheduler(optim_ae, warmup_epochs, args.epochs, min_factor)
        sched_d = build_warmup_scheduler(optim_d, warmup_epochs, args.epochs, min_factor)
        start_epoch = 1
        finetune_mode = True

    # Freeze encoder if requested
    if args.freeze_encoder:
        frozen_model = unwrap_module(model)
        for p in frozen_model.encoder.parameters():
            p.requires_grad = False
        # 仅优化 decoder+final（也可保留判别器）
        optim_ae = torch.optim.AdamW([
            {"params": frozen_model.decoder_core.parameters(), "lr": args.ae_lr, "weight_decay": args.weight_decay},
            {"params": frozen_model.final.parameters(), "lr": args.ae_lr, "weight_decay": args.weight_decay},
        ], lr=args.ae_lr, weight_decay=args.weight_decay)
        # 重建调度器以适配新优化器
        sched_ae = build_warmup_scheduler(optim_ae, warmup_epochs, args.epochs, min_factor)
        print("[Finetune] Encoder frozen. Optimizing decoder_core + final only.")

    # Training loop
    best_ssim = -1.0
    log_rows = []

    total_epochs_to_run = args.epochs
    final_start = start_epoch
    final_end = start_epoch + total_epochs_to_run - 1

    # Directory prefix
    tag = "resume" if resume_mode else ("finetune" if finetune_mode or args.freeze_encoder else "train")
    print(f"Mode: {tag} | Running epochs {final_start}..{final_end}")

    for ep in range(start_epoch, start_epoch + args.epochs):
        weight_sched.step(ep, verbose=(ep % 20 == 0 or ep == 1))
        stats = train_one_epoch(model, disc, train_loader, optim_ae, optim_d, device, cfg, perceptual, modality, loss_l1=w_l1)
        val_stats = evaluate(model, val_loader, device)

        comp_path = save_recon_grid(model, vis_imgs, device, vis_dir, epoch=ep, num_images=8)

        lr_ae = optim_ae.param_groups[0]["lr"]
        lr_d = optim_d.param_groups[0]["lr"]
        print(f"Epoch {ep}/{final_end} | train: loss={stats['loss']:.4f}, l1={stats['l1']:.4f}, "
              f"perc={stats['perc']:.4f}, adv_g={stats['adv_g']:.4f}, adv_d={stats['adv_d']:.4f}, "
              f"ssim={stats['ssim']:.4f} | val: ssim={val_stats['ssim']:.4f}, l1={val_stats['l1']:.4f} | "
              f"LRs -> AE: {lr_ae:.3e}, D: {lr_d:.3e} | vis: {comp_path}")

        log_rows.append({
            "epoch": ep,
            "train_loss": float(stats["loss"]),
            "train_l1": float(stats["l1"]),
            "train_perc": float(stats["perc"]),
            "train_adv_g": float(stats["adv_g"]),
            "train_adv_d": float(stats["adv_d"]),
            "train_ssim": float(stats["ssim"]),
            "val_ssim": float(val_stats["ssim"]),
            "val_l1": float(val_stats["l1"]),
            "lr_ae": float(lr_ae),
            "lr_d": float(lr_d),
        })

        if val_stats["ssim"] > best_ssim:
            best_ssim = val_stats["ssim"]
            save_checkpoint(f"{base_dir}/aae_{modality.lower()}_{tag}_best.pth", model, disc, optim_ae, optim_d, ep, cfg)

        sched_ae.step()
        sched_d.step()

    save_checkpoint(f"{base_dir}/aae_{modality.lower()}_{tag}_last.pth", model, disc, optim_ae, optim_d, ep, cfg)
    save_logs_and_curves(log_rows, out_dir=base_dir, modality=modality)


if __name__ == "__main__":
    main()
