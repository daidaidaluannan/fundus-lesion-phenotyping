# Docker 部署/训练/推理指南

本指南汇总 README 中的 Docker 相关用法，包含镜像构建、Web 推理、命令行推理、训练流程及潜变量导出。按需复制命令并替换为你的本地路径。

## 前置条件
- 已安装 Docker；使用 GPU 时需安装 nvidia-container-runtime，并在命令中加入 `--gpus all`。
- 仓库根目录存在 `fundus_env.tar.gz`（预打包 conda 环境）。若没有，请修改 `Dockerfile` 改为用 `environment.yml` 在线安装（耗时更长）。
- 模型权重自行准备，运行时通过挂载目录传入容器。

## 构建镜像
```bash
docker build -t fundus-lesion-phenotyping .
```

## 路径约定（示例）
- 数据：宿主 `/data/...` 挂载到容器 `/data`：`-v /data:/data`
- 权重：宿主 `/weights` 挂载到容器 `/weights`：`-v /weights:/weights`
- 输出：写回 `/data/outputs`

> 如果本地路径不同，按需替换；容器内命令中的路径需与挂载保持一致。

## Web 推理（可视化界面）
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
```
浏览器访问 http://localhost:8000 上传图像、可选掩膜并查看输出。CPU 运行：去掉 `--gpus all`，并把命令中的 `--device cuda` 改为 `cpu`。

## 命令行推理（无 Web）
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python latent_extract/counterfactual_pipeline.py \
             --image-dir /data/images \
             --mask-dir /data/masks \  # 若无掩膜可省略，脚本会自动生成
             --ret-weights /weights/aae_c.pth \
             --gen-weights /weights/aotgan_netG.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --out-dir /data/outputs/cf_run"
```

## 训练 AAE（CFP 示例）
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
```
OCT 训练：去掉 `--train-masks/--val-masks`，改用 `--modality OCT`。

## 训练 AOT-GAN（可自动掩膜）
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/aotgan_1215.py \
             --train-images /data/train_images \
             --train-masks /data/train_masks \  # 若无掩膜可省略
             --val-images /data/val_images \
             --val-masks /data/val_masks \
             --ret-weights /weights/aae_cfp_best.pth \
             --out-dir /data/outputs/aotgan_run \
             --color-mode color \
             --mask-threshold 0.153"
```

## 伪时序 / 潜变量分析
```bash
docker run --gpus all --rm -it \
  -v /data:/data \
  -v /weights:/weights \
  fundus-lesion-phenotyping \
  bash -c "export PYTHONPATH=/workspace/fundus-lesion-phenotyping && \
           python train_models/Pseudotime_latent.py \
             --data-dir /data/images \
             --csv /data/list.csv \  # 第一列为图像文件名
             --out-dir /data/outputs/pseudotime \
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --multimap-ckpt /weights/aae_s_multimap.pth \
             --color-mode gray \
             --mask-threshold 0.153"
```

## 导出 multimap 潜变量 (export_latent.py)
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
             --multimap-config train_models/multimap_config.json \  # 可选
             --aae-c-ckpt /weights/aae_cfp_best.pth \
             --aot-ckpt /weights/aotgan_gen.pth \
             --color-mode gray \
             --mask-threshold 0.153 \
             --top-k 10"  # 仅保留幅值前 10% 的 latent 元素，默认导出全部
```
参数要点：
- `--image-dir` 必填；输出 CSV 默认为该目录下 `latents.csv`，可用 `--out-csv` 自定义。
- `--color-mode` 与 AOT-GAN 训练模式保持一致（`gray` 或 `color`）。
- `--reduction mean/max` 控制通道聚合，默认 mean -> 7×7=49 个元素；`--top-k` 以百分比保留最大值子集。
- 若用 CPU，去掉 `--gpus all` 并设置 `--device cpu`。

## 常见问题
- 首次运行若 `model_paths.json` 路径不匹配，可删除该文件后重启容器。
- 多 GPU 时 batch size 建议 ≥ GPU 数以启用 DataParallel；否则自动退回单卡。
- 未提供 `fundus_env.tar.gz` 时需调整 Dockerfile 使用 `environment.yml` 在线安装，构建时间会显著增加。
