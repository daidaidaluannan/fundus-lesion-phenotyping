FROM docker.m.daocloud.io/nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ARG ENV_TAR=fundus_env.tar.gz
WORKDIR /workspace/fundus-lesion-phenotyping


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ffmpeg libgl1 libglib2.0-0 ca-certificates unzip python3-pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir conda-pack

COPY ${ENV_TAR} /tmp/env.zip
RUN mkdir -p /opt/conda/envs && \
    unzip /tmp/env.zip -d /opt/conda/envs && \
    ENV_DIR=$(unzip -l /tmp/env.zip | awk '/\/$/ {print $4; exit}' | sed 's:/*$::') && \
    mv "/opt/conda/envs/$ENV_DIR" /opt/conda/envs/fundus && \
    if [ -x /opt/conda/envs/fundus/bin/conda-unpack ]; then \
        /opt/conda/envs/fundus/bin/conda-unpack; \
    else \
        echo "WARNING: conda-unpack not found in packed env; skipping"; \
    fi && \
    rm /tmp/env.zip

ENV PATH=/opt/conda/envs/fundus/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/fundus-lesion-phenotyping

COPY . /workspace/fundus-lesion-phenotyping
CMD ["bash"]