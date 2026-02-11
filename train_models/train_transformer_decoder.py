#!/usr/bin/env python
"""
Train a transformer-based decoder that maps the 1D latent vector from Ret_AAE_0901
into the spatial latent map produced by Ret_AAE_multimap.

Both pretrained encoders remain frozen; only the transformer decoder is updated.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from train_models.AAE_C import AAE as BaselineAAE
from train_models.AAE_C import make_dataloader
from train_models.AAE_S import AAE as MultiMapAAE


class DropPath(nn.Module):
    """Stochastic depth regularization."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop_path_self = DropPath(drop_path)
        self.drop_path_cross = DropPath(drop_path)
        self.drop_path_ffn = DropPath(drop_path)

    def forward(self, spatial_tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        # Spatial self-attention.
        residual = spatial_tokens
        x = self.norm_self(spatial_tokens)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        spatial_tokens = residual + self.drop_path_self(attn_out)

        # Cross attention: let spatial tokens attend to condition tokens derived from z_c.
        residual = spatial_tokens
        x = self.norm_cross(spatial_tokens)
        cross_out, _ = self.cross_attn(x, cond_tokens, cond_tokens, need_weights=False)
        spatial_tokens = residual + self.drop_path_cross(cross_out)

        # Feed-forward network.
        residual = spatial_tokens
        x = self.norm_ffn(spatial_tokens)
        spatial_tokens = residual + self.drop_path_ffn(self.ffn(x))
        return spatial_tokens


class TransformerLatentDecoder(nn.Module):
    """Maps compact latent vectors to spatial latent maps via cross-attentional transformer blocks."""

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        spatial_hw: Tuple[int, int],
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_tokens: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        h, w = spatial_hw
        spatial_tokens = h * w
        self.spatial_hw = spatial_hw
        self.out_channels = out_channels
        self.cond_tokens = cond_tokens
        self.d_model = d_model

        self.cond_proj = nn.Linear(latent_dim, cond_tokens * d_model)
        self.cond_norm = nn.LayerNorm(d_model)

        self.spatial_tokens = nn.Parameter(torch.randn(1, spatial_tokens, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, spatial_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if num_layers <= 0:
            raise ValueError("num_layers must be positive for the transformer decoder")
        if not (d_model % num_heads == 0):
            raise ValueError("d_model must be divisible by num_heads")

        drop_path_rates = torch.linspace(0.0, drop_path, steps=num_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path_rates[idx],
                )
                for idx in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = z.size(0)
        cond = self.cond_proj(z).view(b, self.cond_tokens, self.d_model)
        cond = self.cond_norm(cond)

        spatial = self.spatial_tokens.expand(b, -1, -1) + self.pos_embed
        for block in self.blocks:
            spatial = block(spatial, cond)
        spatial = self.final_norm(spatial)
        logits = self.head(spatial)

        h, w = self.spatial_hw
        logits = logits.view(b, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return logits


def build_multimap_model(
    ckpt_path: Path,
    device: torch.device,
    upsample_override: Optional[str] = None,
) -> Tuple[MultiMapAAE, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    modality = cfg.get("modality", "CFP")
    in_ch = 3 if modality.upper() == "CFP" else 1
    out_ch = in_ch
    num_blocks = cfg.get("num_blocks", 5 if in_ch == 3 else 4)
    base_channels = cfg.get("base_channels", 64)
    latent_channels = cfg.get("latent_channels", 128)
    latent_spatial = cfg.get("latent_spatial")
    if latent_spatial is None:
        img_size = cfg.get("img_size", 224)
        latent_spatial = img_size // (2**num_blocks)
    img_size = cfg.get("img_size", 224)
    upsample_mode = cfg.get("upsample_mode")
    if upsample_override is not None:
        upsample_mode = upsample_override
    if upsample_mode is None:
        upsample_mode = "deconv"

    model = MultiMapAAE(
        in_channels=in_ch,
        out_channels=out_ch,
        num_blocks=num_blocks,
        base_channels=base_channels,
        latent_channels=latent_channels,
        latent_spatial=latent_spatial,
        img_size=img_size,
        upsample_mode=upsample_mode,
    ).to(device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[multimap] missing keys: {missing}")
        print(f"[multimap] unexpected keys: {unexpected}")
    model.eval()
    return model, cfg


def build_baseline_model(
    ckpt_path: Path,
    device: torch.device,
    upsample_override: Optional[str] = None,
) -> Tuple[BaselineAAE, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    modality = cfg.get("modality", "CFP")
    in_ch = 3 if modality.upper() == "CFP" else 1
    out_ch = in_ch
    num_blocks = cfg.get("num_blocks", 5 if in_ch == 3 else 4)
    base_channels = cfg.get("base_channels", 64)
    latent_dim = cfg.get("latent_dim", 256)
    img_size = cfg.get("img_size", 224)
    upsample_mode = cfg.get("upsample_mode")
    if upsample_override is not None:
        upsample_mode = upsample_override
    if upsample_mode is None:
        upsample_mode = "deconv"

    model = BaselineAAE(
        in_channels=in_ch,
        out_channels=out_ch,
        num_blocks=num_blocks,
        base_channels=base_channels,
        latent_dim=latent_dim,
        img_size=img_size,
        upsample_mode=upsample_mode,
    ).to(device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[baseline] missing keys: {missing}")
        print(f"[baseline] unexpected keys: {unexpected}")
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_latents(
    baseline: BaselineAAE,
    multimap: MultiMapAAE,
    imgs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zc_out = baseline.encode(imgs)
    z_c = zc_out[1] if isinstance(zc_out, tuple) else zc_out
    if z_c.dim() > 2:
        z_c = z_c.flatten(1)
    zs_out = multimap.encode(imgs)
    z_s = zs_out[1] if isinstance(zs_out, tuple) else zs_out
    return z_c, z_s


def latent_loss(pred: torch.Tensor, target: torch.Tensor, cosine_weight: float = 0.0) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    if cosine_weight > 0.0:
        b = pred.size(0)
        pred_flat = pred.view(b, -1)
        tgt_flat = target.view(b, -1)
        cos = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()
        return l1 + cosine_weight * cos
    return l1


def train_one_epoch(
    decoder: TransformerLatentDecoder,
    baseline: BaselineAAE,
    multimap: MultiMapAAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cosine_weight: float,
) -> Dict[str, float]:
    decoder.train()
    total_loss = 0.0
    n = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            z_c, z_s = encode_latents(baseline, multimap, imgs)
        pred = decoder(z_c)
        loss = latent_loss(pred, z_s, cosine_weight=cosine_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return {"train_loss": total_loss / max(1, n)}


@torch.no_grad()
def evaluate(
    decoder: TransformerLatentDecoder,
    baseline: BaselineAAE,
    multimap: MultiMapAAE,
    loader: DataLoader,
    device: torch.device,
    cosine_weight: float,
) -> Dict[str, float]:
    decoder.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_cos = 0.0
    n = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        z_c, z_s = encode_latents(baseline, multimap, imgs)
        pred = decoder(z_c)
        total_loss += latent_loss(pred, z_s, cosine_weight=cosine_weight).item() * imgs.size(0)
        total_mae += F.l1_loss(pred, z_s, reduction="none").mean(dim=(1, 2, 3)).sum().item()
        pred_flat = pred.view(pred.size(0), -1)
        z_s_flat = z_s.view(z_s.size(0), -1)
        total_cos += F.cosine_similarity(pred_flat, z_s_flat, dim=1).sum().item()
        n += imgs.size(0)
    return {
        "val_loss": total_loss / max(1, n),
        "val_mae": total_mae / max(1, n),
        "val_cos": total_cos / max(1, n),
    }


def save_checkpoint(
    path: Path,
    decoder: TransformerLatentDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer latent decoder (Z_c -> Z_s)")
    parser.add_argument("--baseline-ckpt", type=str, required=True, help="Ret_AAE_0901 checkpoint path")
    parser.add_argument("--multimap-ckpt", type=str, required=True, help="Ret_AAE_multimap checkpoint path")
    parser.add_argument("--modality", type=str, default="CFP", choices=["CFP", "OCT"])
    parser.add_argument("--train-images", type=str, default="../vae_modle/train_data/global_kaggle/train/normal/")
    parser.add_argument("--val-images", type=str, default="../vae_modle/train_data/global_kaggle/val/normal/")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--baseline-upsampler",
        type=str,
        default=None,
        choices=["deconv", "bilinear", "pixelshuffle"],
        help="Override upsampling mode for baseline decoder if checkpoint cfg is missing.",
    )
    parser.add_argument(
        "--multimap-upsampler",
        type=str,
        default=None,
        choices=["deconv", "bilinear", "pixelshuffle"],
        help="Override upsampling mode for multimap decoder if checkpoint cfg is missing.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1, help="Weight for cosine similarity loss")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="./transformer_decoder_ckpts")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    # Transformer-specific hyperparameters.
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--cond-tokens", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    baseline_model, baseline_cfg = build_baseline_model(
        Path(args.baseline_ckpt),
        device,
        upsample_override=args.baseline_upsampler,
    )
    multimap_model, multimap_cfg = build_multimap_model(
        Path(args.multimap_ckpt),
        device,
        upsample_override=args.multimap_upsampler,
    )

    for p in baseline_model.parameters():
        p.requires_grad_(False)
    for p in multimap_model.parameters():
        p.requires_grad_(False)
    baseline_model.eval()
    multimap_model.eval()

    latent_dim = getattr(baseline_model, "latent_dim", baseline_cfg.get("latent_dim", 256))
    latent_channels = getattr(multimap_model, "latent_channels", multimap_cfg.get("latent_channels", 128))
    latent_spatial = getattr(multimap_model, "latent_spatial", multimap_cfg.get("latent_spatial"))
    if latent_spatial is None:
        img_size = multimap_cfg.get("img_size", args.img_size)
        num_blocks = multimap_cfg.get("num_blocks", 5 if args.modality.upper() == "CFP" else 4)
        latent_spatial = img_size // (2**num_blocks)
    if isinstance(latent_spatial, (tuple, list)):
        spatial_hw = tuple(latent_spatial)
    else:
        spatial_hw = (int(latent_spatial), int(latent_spatial))

    decoder = TransformerLatentDecoder(
        latent_dim=latent_dim,
        out_channels=latent_channels,
        spatial_hw=spatial_hw,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        cond_tokens=args.cond_tokens,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path=args.drop_path,
    ).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    decoder_cfg = {
        "latent_dim": latent_dim,
        "latent_channels": latent_channels,
        "latent_spatial": spatial_hw,
        "modality": args.modality,
        "baseline_ckpt": args.baseline_ckpt,
        "multimap_ckpt": args.multimap_ckpt,
        "baseline_upsampler": args.baseline_upsampler,
        "multimap_upsampler": args.multimap_upsampler,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "cond_tokens": args.cond_tokens,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
        "drop_path": args.drop_path,
        "cosine_weight": args.cosine_weight,
    }

    train_loader = make_dataloader(
        image_dir=args.train_images,
        mask_dir=None,
        modality=args.modality,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        subset_ratio=None,
    )
    val_loader = make_dataloader(
        image_dir=args.val_images,
        mask_dir=None,
        modality=args.modality,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        subset_ratio=None,
    )

    best_loss: Optional[float] = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_log.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_cos,lr\n")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            decoder,
            baseline_model,
            multimap_model,
            train_loader,
            optimizer,
            device,
            cosine_weight=args.cosine_weight,
        )
        val_stats = evaluate(
            decoder,
            baseline_model,
            multimap_model,
            val_loader,
            device,
            cosine_weight=args.cosine_weight,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_stats['train_loss']:.6f} "
            f"val_loss={val_stats['val_loss']:.6f} "
            f"val_mae={val_stats['val_mae']:.6f} "
            f"val_cos={val_stats['val_cos']:.6f} "
            f"lr={lr:.3e}"
        )
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_stats['train_loss']:.6f},{val_stats['val_loss']:.6f},"
                f"{val_stats['val_mae']:.6f},{val_stats['val_cos']:.6f},{lr:.6e}\n"
            )

        if args.save_best:
            if best_loss is None or val_stats["val_loss"] < best_loss:
                best_loss = val_stats["val_loss"]
                save_checkpoint(out_dir / "decoder_best.pth", decoder, optimizer, epoch, decoder_cfg)

        if epoch == args.epochs or (epoch % 10 == 0):
            save_checkpoint(out_dir / f"decoder_epoch_{epoch:03d}.pth", decoder, optimizer, epoch, decoder_cfg)

    save_checkpoint(out_dir / "decoder_last.pth", decoder, optimizer, args.epochs, decoder_cfg)

    decoder.eval()
    with torch.no_grad():
        sample_imgs, _ = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        z_c, z_s = encode_latents(baseline_model, multimap_model, sample_imgs)
        pred = decoder(z_c)
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(exist_ok=True, parents=True)
        for idx in range(min(8, sample_imgs.size(0))):
            target_map = z_s[idx].unsqueeze(0)
            pred_map = pred[idx].unsqueeze(0)
            diff_map = torch.abs(target_map - pred_map)
            save_image(
                normalize_map(pred_map),
                sample_dir / f"sample_{idx:02d}_pred.png",
            )
            save_image(
                normalize_map(target_map),
                sample_dir / f"sample_{idx:02d}_target.png",
            )
            save_image(
                normalize_map(diff_map),
                sample_dir / f"sample_{idx:02d}_diff.png",
            )


def normalize_map(t: torch.Tensor) -> torch.Tensor:
    norm = (t - t.amin(dim=(1, 2, 3), keepdim=True)) / (
        t.amax(dim=(1, 2, 3), keepdim=True) - t.amin(dim=(1, 2, 3), keepdim=True) + 1e-6
    )
    if norm.size(1) == 1:
        return norm
    if norm.size(1) == 3:
        return norm
    heat = norm.mean(dim=1, keepdim=True)
    return heat.repeat(1, 3, 1, 1)


if __name__ == "__main__":
    main()
