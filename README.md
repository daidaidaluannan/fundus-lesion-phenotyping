<p align="center">
<img src="doc/img/logo.png" width="200" class="center">
</p>
<h1 align="center">
Fundus-Lesion-Phenotyping
</h1>

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

## Data & Weights
- CFP runs expect RGB images and optional vessel masks; OCT runs expect single-channel images without masks.
- Prepare folders like `/path/to/train_images`, `/path/to/train_masks`, `/path/to/val_images`, `/path/to/val_masks`.
- Place pretrained AAE weights (e.g., `aae_cfp_train_best.pth`) anywhere and pass the path via `--ret-weights` for AOT-GAN.

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
- Form fields include mask threshold, color mode, Top-K %, latent step/max. Language toggle (EN/中文) in UI.
