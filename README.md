<p align="center">
<img src="doc/img/logo.png" width="200" class="center">
</p>
<h1 align="center">
Fundus-Lesion-Phenotyping
</h1>

<p align="center">
  Language: <a href="README.md">English</a> | <a href="README.zh.md">Chinese</a>
</p>

This folder gathers the key training and inference scripts for the retina AAE and AOT-GAN workflow, plus the `model_zoo/aotgan` dependencies.
<p align="center">
<img src="doc/img/workflow.png">
</p>
<p align="center">
  <em>Workflow of the proposed pipeline</em>
</p>

## Layout
- `train_demo/AAE_C.py`: Retina adversarial autoencoder (CFP/OCT) with latent GAN.
- `train_demo/AAE_S.py`: Multi-map latent AAE variant.
- `train_demo/aotgan_1215.py`: AOT-GAN training that uses a pretrained AAE to generate anomaly masks.
- `train_demo/aotgan_inference_demo.py`, `counterfactual_pipeline.py`: Inference / counterfactual demos.
- `model_zoo/aotgan/`: Generator, discriminator, and loss modules needed by AOT-GAN.
- `PYTHONPATH`: always add the repo root dynamically (e.g., `export PYTHONPATH=$(pwd)`) instead of hard-coding absolute paths; prefer computing `repo_root = Path(__file__).resolve().parent.parent` inside scripts if you move the folder.

## Environment
- Python ≥ 3.9, PyTorch 2.0+, torchvision 0.15+, plus matplotlib, Pillow, tqdm, and standard scientific stack.
- Recommended: create a conda env from `environment.yml`:
  ```bash
  conda env create -f environment.yml
  conda activate fundus-lesion-phenotyping
  ```
  Or install the minimal stack manually: `pip install torch torchvision matplotlib pillow tqdm`
- Add the project root to `PYTHONPATH` when running from this bundle:
  ```bash
  cd release_bundle
  export PYTHONPATH=$(pwd)
  ```
  The training/inference scripts compute `repo_root = Path(__file__).resolve().parent.parent` to avoid hard-coded paths; ensure `PYTHONPATH` includes the repo root if you move files around.

## Docker Quickstart
- Requirements: Docker installed. For GPU, install `nvidia-container-runtime` and add `--gpus all` to `docker run`. The image expects a bundled `fundus_env.tar.gz`; if missing, adjust `Dockerfile` to build from `environment.yml` (slower).
- Build the image:
  ```bash
  docker build -t fundus-lesion-phenotyping .
  ```
- Mounting example: map data to `/data` and weights to `/weights` so container paths match the commands (`-v /data:/data -v /weights:/weights`).

**Web inference (FastAPI UI)**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  -p 8000:8000 \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python web/run_server.py \
             --aae-c /weights/aae_c.pth \
             --aot-gan /weights/aotgan_netG.pth \
             --aae-s /weights/aae_s_multimap.pth \
             --device cuda \
             --host 0.0.0.0 --port 8000"
# Open http://localhost:8000. For CPU, drop --gpus all and set --device cpu.
```

**CLI inference (no Web UI)**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python latent_extract/counterfactual_pipeline.py \
             --image-dir /data/images \
             --mask-dir /data/masks \  # optional; auto-generated if omitted
             --ret-weights /weights/aae_c.pth \
             --gen-weights /weights/aotgan_netG.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --out-dir /data/outputs/cf_run"
```

**Train AAE (CFP example)**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/AAE_C.py \
             --modality CFP \
             --train-images /data/train_images \
             --train-masks /data/train_masks \
             --val-images /data/val_images \
             --val-masks /data/val_masks \
             --out-dir /data/outputs/aae_cfp"
# OCT: drop mask args and set --modality OCT.
```

**Train AOT-GAN (auto masks supported)**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/aotgan_1215.py \
             --train-images /data/train_images \
             --train-masks /data/train_masks \  # optional
             --val-images /data/val_images \
             --val-masks /data/val_masks \
             --ret-weights /weights/aae_cfp_best.pth \
             --out-dir /data/outputs/aotgan_run \
             --color-mode color \
             --mask-threshold 0.153"
```

**Pseudotime / latent analysis**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/Pseudotime_latent.py \
             --data-dir /data/images \
             --csv /data/list.csv \  # first column: image filenames
             --out-dir /data/outputs/pseudotime \
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --color-mode gray \
             --mask-threshold 0.153"
```

**Export multimap latents**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python latent_extract/export_latent.py \
             --image-dir /data/images \
             --out-csv /data/outputs/latents.csv \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --multimap-config train_models/multimap_config.json \  # optional
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --color-mode gray \
             --mask-threshold 0.153 \
             --top-k 10"  # keep top 10% by magnitude
```

**Tips**
- If `web/model_paths.json` contains stale absolute paths, delete it and restart the container.
- With multiple GPUs, prefer batch size ≥ GPU count to enable DataParallel; otherwise it falls back to single GPU.
- Without the pre-bundled env, edit `Dockerfile` to install from `environment.yml`, noting the longer build time.

## Data & Weights
- CFP runs expect RGB images and optional vessel masks; OCT runs expect single-channel images without masks.
- Prepare folders like `/path/to/train_images`, `/path/to/train_masks`, `/path/to/val_images`, `/path/to/val_masks`.
- Place pretrained AAE weights (e.g., `aae_cfp_train_best.pth`) anywhere and pass the path via `--ret-weights` for AOT-GAN.
- Pretrained checkpoints (AAE_C, AAE_S, AOT-GAN) are available via Baidu NetDisk: link `https://pan.baidu.com/s/156_qJ-iV3WttUverDVQmQw` with extraction code `c2mv`. Download `model_ckp` and mount or copy into your run.
- Environment archive (`env.zip`) via Baidu NetDisk: link `https://pan.baidu.com/s/1e3RapQiSFr-fg57thA5t_g` with extraction code `k6wr`. Unzip and follow instructions inside, or build from `environment.yml` as an alternative.

## Train AAE (CFP example)
```bash
cd release_bundle
export PYTHONPATH=$(pwd)
python train_demo/AAE_C.py \
  --modality CFP \
  --train-images /path/to/train_images \
  --train-masks /path/to/train_masks \
  --val-images /path/to/val_images \
  --val-masks /path/to/val_masks \
  --out-dir outputs/aae_cfp
```
- For OCT, drop the mask args and use `--modality OCT`.
- `AAE_S.py` trains the multi-map latent variant with similar arguments (see defaults inside for dataset paths).

## Train AOT-GAN with auto masks
```bash
cd release_bundle
export PYTHONPATH=$(pwd)
python train_demo/aotgan_1215.py \
  --train-images /path/to/train_images \
  --train-masks /path/to/train_masks \  # optional; if omitted, masks are auto-generated
  --val-images /path/to/val_images \
  --val-masks /path/to/val_masks \
  --ret-weights /path/to/aae_cfp_train_best.pth \
  --out-dir outputs/aotgan_run \
  --color-mode color \
  --mask-threshold 0.153
```
- If you only have images, omit mask flags; the script will generate masks from the AAE reconstruction (save location controlled by `--mask-dir`).
- For simple folder mode (no split), use `--image-dir /path/to/images` instead of `--train-images`.

## Inference / Counterfactuals
- `aotgan_inference_demo.py`: load trained AOT-GAN weights and run inpainting on test images. Typical args: `--image-dir`, `--gen-weights`, `--mask-dir` (or auto-mask with a pretrained AAE).
- `counterfactual_pipeline.py`: end-to-end pipeline that (1) loads pretrained AAE/AOT-GAN, (2) generates or loads masks, (3) runs inpainting, and (4) produces counterfactual visualizations/exports. Key flags mirror the training scripts: point `--ret-weights` to the AAE checkpoint, `--gen-weights` to the AOT-GAN generator, and provide `--image-dir`/`--mask-dir` (or rely on auto-masks). Check the script docstring for optional exports and thresholds.
- `Pseudotime_latent.py`: builds a diffusion pseudotime axis from latent anomaly features. It uses pretrained AAE_C (for coarse recon), AOT-GAN (for masks/inpainting), and AAE_S (multi-map) to extract latent differences, then runs kNN + diffusion maps (and optional MST) to output pseudotime CSV/NPZ and UMAP/t-SNE plots. Typical usage:
  ```bash
  cd release_bundle
  export PYTHONPATH=$(pwd)
  python train_demo/Pseudotime_latent.py \
    --data-dir /path/to/images \
    --csv /path/to/list.csv \  # first column: image names
    --out-dir outputs/pseudotime \
    --aae-c-ckpt /path/to/aae_cfp_train_best.pth \
    --aot-ckpt /path/to/aotgan_gen.pth \
    --multimap-ckpt /path/to/aae_s_multimap.pth \
    --color-mode gray \
    --mask-threshold 0.153
  ```

## Tips
- Use `--progress` to enable tqdm progress bars where available.
- Multi-GPU: scripts support DataParallel when batch size ≥ GPU count; otherwise they fall back to single GPU.
- Logging: training scripts write CSV logs and PNG curves/visualizations to the `--out-dir` you specify.

## Web Demo (FastAPI)
- Located at `release_bundle/web/`. Start server via CLI (models from args):
  ```bash
  cd release_bundle
  export PYTHONPATH=$(pwd)
  python web/run_server.py \
    --aae-c /path/to/aae_c.pth \
    --aot-gan /path/to/aotgan_netG.pth \
    --aae-s /path/to/aae_s_multimap.pth \
    --device cuda \
    --host 0.0.0.0 --port 8000 --reload
  ```
- `model_paths.json` is generated by `run_server.py` to persist the last used paths; if you see absolute paths from another machine, delete or overwrite the file before committing, and always pass fresh paths via CLI flags.
- Open `http://localhost:8000/` to upload an image and supply weight paths (`AAE_C`, `AOT-GAN netG`, `AAE_S`). Returns:
  - Input, coarse recon, pseudo-healthy, auto-mask
  - Latent overlays (mean/max), latent grid
  - Optional latent adjustment panel (scales) and Top-K latent transfer result
- Form fields include mask threshold, color mode, Top-K %, latent step/max. Language toggle (EN/Chinese) in UI.
