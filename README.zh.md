<p align="center">
<img src="doc/img/logo.png" width="200" class="center">
</p>
<h1 align="center">
Fundus-Lesion-Phenotyping
</h1>

<p align="center">
  语言：<a href="README.md">English</a> | <a href="README.zh.md">中文</a>
</p>

本仓库提供视网膜 AAE 与 AOT-GAN 的训练、推理与可视化代码，以及所需的 `model_zoo/aotgan` 依赖。
<p align="center">
<img src="doc/img/workflow.png">
</p>
<p align="center">
  <em>整体工作流示意</em>
</p>

## 目录结构
- `train_models/AAE_C.py`：彩色 CFP / 灰度 OCT 的对抗自编码器，带潜空间判别器与可选感知损失。
- `train_models/AAE_S.py`：多映射 latent 版本，供后续可视化与迁移。
- `train_models/aotgan_1215.py`：利用预训练 AAE 自动生成掩膜的 AOT-GAN 训练脚本。
- `latent_extract/`：命令行推理、反事实生成、潜变量导出与可视化工具。
- `model_zoo/aotgan/`：AOT-GAN 生成器、判别器、损失等模块。
- `web/`：FastAPI 前端，提供上传/可视化界面。

## 环境
- Python ≥ 3.9，PyTorch 2.0+，torchvision 0.15+，以及 matplotlib、Pillow、tqdm 等常见科学计算库。
- 推荐使用 conda：
  ```bash
  conda env create -f environment.yml
  conda activate fundus-lesion-phenotyping
  ```
- 运行前确保将仓库根目录加入 `PYTHONPATH`：
  ```bash
  export PYTHONPATH=$(pwd)
  ```

## 数据与权重
- CFP 需要 RGB 图像和可选血管掩膜；OCT 为单通道，无需掩膜。
- 准备训练/验证图像与掩膜文件夹（文件名一一对应）；也可只提供图像，AOT-GAN 会用 AAE 自动生成掩膜。
- 预训练权重路径在运行时通过 CLI 传入。
- 预训练权重（AAE_C、AAE_S、AOT-GAN）百度网盘：`https://pan.baidu.com/s/156_qJ-iV3WttUverDVQmQw`，提取码 `c2mv`，文件夹名 `model_ckp`。
- 环境压缩包 `env.zip` 百度网盘：`https://pan.baidu.com/s/1e3RapQiSFr-fg57thA5t_g`，提取码 `k6wr`。可解压使用，或直接按 `environment.yml` 重新创建环境。

## Docker 快速上手
- 依赖：已安装 Docker；使用 GPU 时需要 `nvidia-container-runtime` 并在 `docker run` 增加 `--gpus all`。镜像默认使用已打包的 `fundus_env.tar.gz`；若缺失可改用 `environment.yml` 在线安装（构建时间更长）。
- 构建镜像：
  ```bash
  docker build -t fundus-lesion-phenotyping .
  ```
- 挂载示例：数据 `/data`，权重 `/weights`，命令中的路径需与挂载一致。

**Web 推理（FastAPI）**
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
# 浏览器访问 http://localhost:8000；CPU 运行时去掉 --gpus all，--device 设置为 cpu。
```

**命令行推理（无 Web）**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python latent_extract/counterfactual_pipeline.py \
             --image-dir /data/images \
             --mask-dir /data/masks \\  # 可省略，脚本可自动生成掩膜
             --ret-weights /weights/aae_c.pth \
             --gen-weights /weights/aotgan_netG.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --out-dir /data/outputs/cf_run"
```

**训练 AAE（CFP 示例）**
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
# OCT：去掉掩膜参数，改用 --modality OCT。
```

**训练 AOT-GAN（支持自动掩膜）**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/aotgan_1215.py \
             --train-images /data/train_images \
             --train-masks /data/train_masks \\  # 无掩膜可省略
             --val-images /data/val_images \
             --val-masks /data/val_masks \
             --ret-weights /weights/aae_cfp_best.pth \
             --out-dir /data/outputs/aotgan_run \
             --color-mode color \
             --mask-threshold 0.153"
```

**伪时序 / 潜变量分析**
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/Pseudotime_latent.py \
             --data-dir /data/images \
             --csv /data/list.csv \\  # 第一列为图像文件名
             --out-dir /data/outputs/pseudotime \
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --color-mode gray \
             --mask-threshold 0.153"
```

**导出 multimap 潜变量**
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
             --multimap-config train_models/multimap_config.json \\  # 可选
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --color-mode gray \
             --mask-threshold 0.153 \
             --top-k 10"  # 仅保留幅值前 10% 的 latent 元素
```

**提示**
- 首次运行若 `web/model_paths.json` 路径不匹配，可删除后重启容器。
- 多 GPU 时 batch size 建议 ≥ GPU 数以启用 DataParallel；否则自动退回单卡。
- 若缺少预打包环境，调整 `Dockerfile` 使用 `environment.yml` 在线安装，构建时间会显著增加。
