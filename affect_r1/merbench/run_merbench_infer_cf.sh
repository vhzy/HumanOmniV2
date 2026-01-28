#!/bin/bash
# 因果干预推理实验本地运行脚本
# set -euo pipefail

# 模型配置
MODEL_PATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo_logit2
PROCESSOR_PATH=$MODEL_PATH
# AFFECTGPT_ROOT=/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo_logit2/inference-cf
CKPT_NAME=checkpoint-3000
DATASETS="OVMERDPlus,MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2" #,MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2
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
EXPERIMENT="early_answer"
MASK_MODALITY="none"
EARLY_ANSWER="--early-answer"

# 实验6: Early Answer + Mask全部模态
# EXPERIMENT="early_answer_mask_all"
# MASK_MODALITY="all"
# EARLY_ANSWER="--early-answer"

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
RUN_NAME="merbench_cf_old_10_${EXPERIMENT}"

# ====================================================

echo "=========================================="
echo "因果干预推理实验配置"
echo "=========================================="
echo "实验类型: ${EXPERIMENT}"
echo "Mask模态: ${MASK_MODALITY}"
echo "Mask比例: ${MASK_RATIO}"
echo "Early Answer: ${EARLY_ANSWER:-disabled}"
echo "运行名称: ${RUN_NAME}"
echo "输出目录: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.run_inference_counterfactual \
  --model-path "${MODEL_PATH}" \
  --processor-path "${PROCESSOR_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --checkpoint-name "${CKPT_NAME}" \
  --datasets "${DATASETS}" \
  --max-new-tokens 1024 \
  --temperature 0.9 \
  --prompt-mode default \
  --top-p 0.9 \
  --do-sample \
  --use-audio-in-video \
  --mask-modality "${MASK_MODALITY}" \
  --mask-ratio ${MASK_RATIO} \
  ${MASK_NOISE} \
  ${EARLY_ANSWER}

echo ""
echo "=========================================="
echo "推理完成！"
echo "结果保存在: ${OUTPUT_ROOT}/${RUN_NAME}/${CKPT_NAME}/"
echo "=========================================="

