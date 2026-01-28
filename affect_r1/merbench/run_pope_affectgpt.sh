#!/bin/bash
# AffectGPT POPE 压力测试一键脚本
#
# 使用示例：
# 测试音频幻觉：
# CUDA_VISIBLE_DEVICES=0 bash run_pope_affectgpt.sh audio
#
# 测试视觉幻觉：
# CUDA_VISIBLE_DEVICES=1 bash run_pope_affectgpt.sh visual
#
# 指定 epoch 和输出目录：
# CUDA_VISIBLE_DEVICES=0 CKPT_EPOCH=30 OUTPUT_DIR=/path/to/output bash run_pope_affectgpt.sh audio

set -e

# 参数
MASK_TYPE=${1:-"audio"}  # audio 或 visual
CKPT_EPOCH=${CKPT_EPOCH:-60}
GPU=${GPU:-0}

# 路径配置
AFFECTGPT_ROOT="/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT"
CFG_PATH="${AFFECTGPT_ROOT}/train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# POPE 数据集路径
POPE_DATA_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset"
if [ "$MASK_TYPE" == "audio" ]; then
    POPE_DATASET="${POPE_DATA_ROOT}/pope_audio_filtered_v2.jsonl"
elif [ "$MASK_TYPE" == "visual" ]; then
    POPE_DATASET="${POPE_DATA_ROOT}/pope_visual_filtered_v2.jsonl"
else
    echo "Error: MASK_TYPE must be 'audio' or 'visual'"
    exit 1
fi

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"${AFFECTGPT_ROOT}/output/pope_results"}
OUTPUT_FILE="${OUTPUT_DIR}/pope_${MASK_TYPE}_epoch${CKPT_EPOCH}.jsonl"

echo "=============================================="
echo "AffectGPT POPE 压力测试"
echo "=============================================="
echo "Mask类型: $MASK_TYPE"
echo "Checkpoint Epoch: $CKPT_EPOCH"
echo "配置文件: $CFG_PATH"
echo "POPE数据集: $POPE_DATASET"
echo "输出文件: $OUTPUT_FILE"
echo "GPU: $GPU"
echo "=============================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行推理
cd "$AFFECTGPT_ROOT"
python "${SCRIPT_DIR}/run_pope_inference_affectgpt.py" \
    --cfg-path "$CFG_PATH" \
    --ckpt-epoch "$CKPT_EPOCH" \
    --pope-dataset "$POPE_DATASET" \
    --output "$OUTPUT_FILE" \
    --mask-type "$MASK_TYPE" \
    --gpu "$GPU" \
    --resume

echo ""
echo "=============================================="
echo "完成！"
echo "结果文件: $OUTPUT_FILE"
echo "=============================================="

