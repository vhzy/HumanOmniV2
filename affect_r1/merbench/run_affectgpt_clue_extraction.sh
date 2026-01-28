#!/bin/bash
# AffectGPT 线索提取一键脚本
#
# 功能：
# 1. 将 AffectGPT NPZ 输出转换为 JSONL 格式
# 2. 运行 check_hallucination.py 提取多模态线索
#
# 使用示例：
# bash run_affectgpt_clue_extraction.sh \
#     /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/ovmerd_description/checkpoint_000030_loss_0.602.npz
#
# 或指定 GPU 和引擎：
# CUDA_VISIBLE_DEVICES=0 ENGINE=qwen bash run_affectgpt_clue_extraction.sh \
#     /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/ovmerd_description/checkpoint_000030_loss_0.602.npz

set -e

# 参数
NPZ_PATH=${1:-"/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/ovmerd_description/checkpoint_000030_loss_0.602.npz"}
ENGINE=${ENGINE:-"qwen"}  # gpt 或 qwen
LLM_NAME=${LLM_NAME:-"Qwen25"}
DATASET_ROOT=${DATASET_ROOT:-"/mnt/afs/hanzhiyuan/MER-UniBench/data"}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 输出路径
JSONL_PATH="${NPZ_PATH%.npz}.jsonl"
OUTPUT_DIR="$(dirname "$NPZ_PATH")"
CLUE_OUTPUT="${OUTPUT_DIR}/clue_extraction_${ENGINE}.jsonl"

echo "=============================================="
echo "AffectGPT Clue Extraction Pipeline"
echo "=============================================="
echo "Input NPZ: $NPZ_PATH"
echo "Engine: $ENGINE"
echo "Output JSONL: $JSONL_PATH"
echo "Clue Output: $CLUE_OUTPUT"
echo "=============================================="

# Step 1: 转换 NPZ 到 JSONL
echo ""
echo "[Step 1/2] Converting NPZ to JSONL..."
python "${SCRIPT_DIR}/convert_affectgpt_npz_to_jsonl.py" \
    --input "$NPZ_PATH" \
    --output "$JSONL_PATH" \
    --mask-modality none  # baseline 模式

# Step 2: 运行线索提取
echo ""
echo "[Step 2/2] Extracting multimodal clues..."
python "${SCRIPT_DIR}/check_hallucination.py" \
    --input "$JSONL_PATH" \
    --output "$CLUE_OUTPUT" \
    --engine "$ENGINE" \
    --llm-name "$LLM_NAME" \
    --dataset-root "$DATASET_ROOT"

echo ""
echo "=============================================="
echo "Done!"
echo "Clue extraction results: $CLUE_OUTPUT"
echo "Statistics report: ${OUTPUT_DIR}/perception_hallucination_${ENGINE}.txt"
echo "=============================================="

