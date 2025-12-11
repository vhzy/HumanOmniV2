#!/bin/bash
# 创建一个干净的、开发机和容器共享的 Conda 环境
# 路径：/mnt/afs/hanzhiyuan/.conda/envs/affect_r1
# 版本与 humanomni 环境保持一致

ENV_NAME="humanomni_v2"
ENV_PATH="/mnt/afs/hanzhiyuan/.conda/envs/${ENV_NAME}"

echo "=== Creating new environment: ${ENV_NAME} ==="

# 使用 --copy 参数确保所有文件都是复制而非硬链接
# 使用 Python 3.12 匹配 humanomni
conda create -p ${ENV_PATH} python=3.12 -y --copy

echo "=== Activating environment ==="
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

# 获取当前环境的标准 site-packages 路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "=== Target Site-Packages: ${SITE_PACKAGES} ==="

echo "=== Installing PyTorch 2.9.1 (CUDA 12.8, matching humanomni) ==="
pip install torch==2.9.1 torchvision==0.24.1 torchaudio --index-url https://download.pytorch.org/whl/cu128 --target ${SITE_PACKAGES} --no-cache-dir --upgrade

echo "=== Installing transformers and related (matching humanomni versions) ==="
pip install transformers==4.57.3 accelerate==1.12.0 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Installing training frameworks (matching humanomni versions) ==="
pip install deepspeed==0.18.2 trl==0.25.1 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Installing data processing (matching humanomni versions) ==="
pip install datasets==4.4.1 pandas==2.3.3 numpy==2.0.0 pillow==12.0.0 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Installing video/audio processing (matching humanomni versions) ==="
pip install decord==0.6.0 av==16.0.1 librosa==0.11.0 soundfile==0.13.1 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Installing Qwen utilities (matching humanomni versions) ==="
pip install qwen-vl-utils==0.0.14 qwen-omni-utils==0.0.8 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Installing other dependencies from humanomni ==="
pip install einops==0.8.1 scipy==1.16.3 pyyaml==6.0.3 safetensors==0.7.0 -i https://mirrors.aliyun.com/pypi/simple/ --target ${SITE_PACKAGES} --no-cache-dir

echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import qwen_omni_utils; print('Qwen Omni Utils: OK')"

echo "=== Environment created successfully ==="
echo "Path: ${ENV_PATH}"
echo "Activation command: conda activate ${ENV_PATH}"

