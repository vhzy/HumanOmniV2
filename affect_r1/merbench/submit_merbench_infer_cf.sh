#!/usr/bin/env bash

# 因果干预推理实验提交脚本
# 支持mask多模态信息和early answer模式

# 提交任务信息
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
PARTITION=amplarge2
# PARTITION=h100-share2
# PARTITION=h100-share3
# PARTITION=m-train-1
# PARTITION=m-train-2
# PARTITION=m-train-ocr
# PARTITION=vqalarge2
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs

nodes=1
GPUS=1

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iq.I10"
else
    DEVICE="N6lS.Iu.I80"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKDIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d_%H%M%S)

# Environment setup
ENV_COMMAND="source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && \
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2"

# Model configuration
MODEL_PATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_all2_v2/checkpoint-2000
PROCESSOR_PATH=$MODEL_PATH
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_all2_v2/checkpoint-2000/inference
CKPT_NAME=checkpoint-3262
DATASETS="OVMERDPlus,MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2"
# DATASETS="IEMOCAPFour"

# ==================== 因果干预配置 ====================
# 选择实验类型（取消注释想要运行的实验）

# 实验1: Baseline (无干预)
# EXPERIMENT="baseline"
# MASK_MODALITY="none"
# EARLY_ANSWER=""

# 实验2: Mask 90% 全部模态
# EXPERIMENT="mask_all"
# MASK_MODALITY="all"
# EARLY_ANSWER=""

# 实验3: Mask 90% 视频信息
# EXPERIMENT="mask_visual"
# MASK_MODALITY="visual"
# EARLY_ANSWER=""

# 实验4: Mask 90% 音频信息
# EXPERIMENT="mask_audio"
# MASK_MODALITY="audio"
# EARLY_ANSWER=""

# 实验5: Early Answer (无mask)
# EXPERIMENT="early_answer"
# MASK_MODALITY="none"
# EARLY_ANSWER="--early-answer"

# 实验6: Early Answer + Mask全部模态
EXPERIMENT="early_answer_mask_all"
MASK_MODALITY="all"
EARLY_ANSWER="--early-answer"

# 实验7: Early Answer + Mask视频
# EXPERIMENT="early_answer_mask_visual"
# MASK_MODALITY="visual"
# EARLY_ANSWER="--early-answer"

# 实验8: Early Answer + Mask音频
# EXPERIMENT="early_answer_mask_audio"
# MASK_MODALITY="audio"
# EARLY_ANSWER="--early-answer"

# Masking configuration
MASK_RATIO=1.0  # 90% masking
MASK_NOISE=""   # Use zeros (add --mask-noise to use noise)

# Run name based on experiment
RUN_NAME="merbench_cf_${EXPERIMENT}"
LOG_FILE="$LOG_DIR/merbench_cf_${EXPERIMENT}_${DATE}.log"

# ====================================================

# Inference command
EXE_COMMAND="cd ${PROJECT_ROOT} && \
export PYTHONPATH=${REPO_ROOT}:\$PYTHONPATH && \
CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.run_inference_counterfactual \
  --model-path ${MODEL_PATH} \
  --processor-path ${PROCESSOR_PATH} \
  --dataset-root ${DATASET_ROOT} \
  --output-root ${OUTPUT_ROOT} \
  --run-name ${RUN_NAME} \
  --checkpoint-name ${CKPT_NAME} \
  --datasets \"${DATASETS}\" \
  --max-new-tokens 1024 \
  --temperature 0.9 \
  --top-p 0.9 \
  --do-sample \
  --use-audio-in-video \
  --mask-modality ${MASK_MODALITY} \
  --mask-ratio ${MASK_RATIO} \
  ${MASK_NOISE} \
  ${EARLY_ANSWER}"

COMMAND="cd \"$WORKDIR\" && ${ENV_COMMAND} && ${EXE_COMMAND} >> \"${LOG_FILE}\" 2>&1"

echo "=========================================="
echo "因果干预推理实验配置"
echo "=========================================="
echo "实验类型: ${EXPERIMENT}"
echo "Mask模态: ${MASK_MODALITY}"
echo "Mask比例: ${MASK_RATIO}"
echo "Early Answer: ${EARLY_ANSWER:-disabled}"
echo "运行名称: ${RUN_NAME}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

# echo ">>> 将在3小时后提交任务"
# echo "Command: ${COMMAND}"
# echo "睡眠3小时..."
# sleep 3h

echo ">>> 正在提交因果干预推理任务"
sco acp jobs create \
--workspace-name "$WORKSPACE" \
-p "$PARTITION" \
--container-image-url "$CONTAINTER" \
--storage-mount "$MOUNT" \
--training-framework pytorch \
--worker-spec "${DEVICE}.${GPUS}" \
--worker-nodes "$nodes" \
--job-name "affect_r1_merbench_cf_${EXPERIMENT}" \
--command "$COMMAND"

