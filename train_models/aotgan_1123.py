#!/usr/bin/env python
"""
Variant of the AOT-GAN training demo that preserves RGB visualizations and
adds checkpoint saving (best/latest).
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image
from PIL import Image
import torch.nn as nn
from torch.nn.utils import spectral_norm

repo_root = Path(__file__).resolve().parent.parent
import sys

if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from model_zoo.aotgan.aotgan import InpaintGenerator, Discriminator
from model_zoo.aotgan.loss import loss as loss_module
from model_zoo.phanes import AnomalyMap
from transforms.synthetic import GenerateMasks
from train_models.AAE_C  import AAE


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, img_size: int = 224):
        self.image_paths = sorted([
            image_dir / fname for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


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


def generate_mask(ano: AnomalyMap,
                  mask_generator,
                  x_gray: torch.Tensor,
                  coarse_gray: torch.Tensor,
                  mask_threshold: float,
                  add_synthetic: bool = True):
    residual, saliency = ano.compute_residual(coarse_gray, x_gray)
    mask_auto = ano.filter_anomaly_mask(residual * saliency,
                                        masking_threshold=mask_threshold)
    if add_synthetic and mask_generator is not None:
        mask_np = mask_generator(x_gray.detach().cpu())
        mask_syn = torch.as_tensor(mask_np, device=x_gray.device, dtype=x_gray.dtype)
        mask = torch.clamp(mask_auto + mask_syn, max=1.0)
    else:
        mask = mask_auto
    return mask


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


def visualize_rgb(netG,
                  ret_aae,
                  ano,
                  mask_generator,
                  samples_rgb: torch.Tensor,
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
        mask = generate_mask(ano, mask_generator, x_gray, coarse_gray,
                             mask_threshold, add_synthetic=False)
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
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optim_G": optim_G.state_dict(),
        "optim_D": optim_D.state_dict(),
        "stats": stats,
    }, path)
    print(f"[Checkpoint] saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="AOT-GAN demo with RGB visualization + checkpoints")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--ret-weights", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.153)
    parser.add_argument("--vis-interval", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "train_demo/aotgan_vis_rgb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-mode", type=str, choices=["gray", "color"], default="gray",
                        help="Use grayscale (original behavior) or RGB training for AOT-GAN.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(Path(args.image_dir))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, pin_memory=True)

    vis_count = min(4, len(dataset))
    vis_samples = torch.stack([dataset[i] for i in range(vis_count)])

    ret_aae = load_ret_aae(Path(args.ret_weights), device)
    netG, netD = build_aotgan_networks(args.color_mode, device)
    optim_G = torch.optim.Adam(netG.parameters(), lr=5e-5, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))
    rec_loss_config = {"L1": 1.0, "Style": 250.0, "Perceptual": 0.1}
    rec_loss_dict = {name: getattr(loss_module, name)() for name in rec_loss_config}
    adv_loss = loss_module.smgan()
    ano = AnomalyMap()
    mask_generator = GenerateMasks(min_size=20, max_size=40)

    best_rec = float("inf")
    stats = {}

    for epoch in range(1, args.epochs + 1):
        netG.train()
        ret_aae.eval()
        epoch_stats = {"rec": 0.0, "gen": 0.0, "dis": 0.0, "n": 0}

        for batch in loader:
            rgb = batch.to(device)
            with torch.no_grad():
                coarse_rgb, _ = ret_aae(rgb)
            x_gray = TF.rgb_to_grayscale(rgb)
            coarse_gray = TF.rgb_to_grayscale(coarse_rgb)
            mask = generate_mask(ano, mask_generator, x_gray, coarse_gray,
                                 mask_threshold=args.mask_threshold, add_synthetic=True)

            rec_l, gen_l, dis_l, _ = train_aot_step(
                rgb, x_gray, mask, netG, netD, optim_G, optim_D,
                rec_loss_config, rec_loss_dict, adv_loss, args.color_mode)
            bsz = x_gray.size(0)
            epoch_stats["rec"] += rec_l * bsz
            epoch_stats["gen"] += gen_l * bsz
            epoch_stats["dis"] += dis_l * bsz
            epoch_stats["n"] += bsz

        n = max(epoch_stats["n"], 1)
        avg_rec = epoch_stats["rec"] / n
        print(f"[Epoch {epoch}/{args.epochs}] rec={avg_rec:.4f}, "
              f"gen={epoch_stats['gen']/n:.4f}, dis={epoch_stats['dis']/n:.4f}")

        stats = {
            "rec": avg_rec,
            "gen": epoch_stats["gen"]/n,
            "dis": epoch_stats["dis"]/n,
            "color_mode": args.color_mode,
        }

        ckpt_dir = out_dir / "checkpoints"
        save_checkpoint(ckpt_dir / "latest.pth", netG, netD, optim_G, optim_D, epoch, stats)
        if avg_rec < best_rec:
            best_rec = avg_rec
            save_checkpoint(ckpt_dir / "best.pth", netG, netD, optim_G, optim_D, epoch, stats)

        if epoch % args.vis_interval == 0:
            visualize_rgb(netG, ret_aae, ano, mask_generator, vis_samples,
                          device, out_dir, epoch, args.mask_threshold, args.color_mode)


if __name__ == "__main__":
    main()
