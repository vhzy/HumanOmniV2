#!/bin/bash
# POPE压力测试实验脚本
# 
# 实验流程：
# 1. 构建POPE问题对数据集
# 2. 运行模型推理（分别mask音频和视觉）
# 3. 评估结果
#
# 使用示例：
# bash run_pope_experiment.sh /path/to/model/checkpoint

set -e

# ==================== 配置 ====================
MODEL_PATH="${1:-/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_papo5_v2/checkpoint-3000}"
GT_CLUES="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl"
VIDEO_ROOT="/mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video"
POPE_DATA_DIR="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset"
OUTPUT_DIR="${MODEL_PATH}/pope_results"

# 获取checkpoint名称
CHECKPOINT_NAME=$(basename "$MODEL_PATH")

echo "=========================================="
echo "POPE压力测试实验"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "GT线索: $GT_CLUES"
echo "视频目录: $VIDEO_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo ""

# ==================== Step 1: 构建POPE数据集 ====================
echo "[Step 1/4] 构建POPE问题对数据集..."
cd /mnt/afs/hanzhiyuan/code/HumanOmniV2

if [ ! -f "$POPE_DATA_DIR/pope_audio.jsonl" ] || [ ! -f "$POPE_DATA_DIR/pope_visual.jsonl" ]; then
    python -m affect_r1.merbench.build_pope_dataset \
        --gt-clues "$GT_CLUES" \
        --video-root "$VIDEO_ROOT" \
        --output-dir "$POPE_DATA_DIR" \
        --max-cues-per-sample 2
else
    echo "  POPE数据集已存在，跳过构建"
    echo "  Audio: $POPE_DATA_DIR/pope_audio.jsonl"
    echo "  Visual: $POPE_DATA_DIR/pope_visual.jsonl"
fi

# ==================== Step 2: 运行Audio POPE推理 ====================
echo ""
echo "[Step 2/4] 运行Audio POPE推理 (mask音频，问模型是否听到音频线索)..."
mkdir -p "$OUTPUT_DIR"

AUDIO_RESULT="$OUTPUT_DIR/pope_audio_results.jsonl"
if [ ! -f "$AUDIO_RESULT" ]; then
    python -m affect_r1.merbench.run_pope_inference \
        --model-path "$MODEL_PATH" \
        --pope-dataset "$POPE_DATA_DIR/pope_audio.jsonl" \
        --output "$AUDIO_RESULT" \
        --mask-type audio \
        --mask-ratio 0.9 \
        --max-new-tokens 50
else
    echo "  Audio POPE结果已存在，跳过推理"
fi

# ==================== Step 3: 运行Visual POPE推理 ====================
echo ""
echo "[Step 3/4] 运行Visual POPE推理 (mask视频，问模型是否看到视觉线索)..."

VISUAL_RESULT="$OUTPUT_DIR/pope_visual_results.jsonl"
if [ ! -f "$VISUAL_RESULT" ]; then
    python -m affect_r1.merbench.run_pope_inference \
        --model-path "$MODEL_PATH" \
        --pope-dataset "$POPE_DATA_DIR/pope_visual.jsonl" \
        --output "$VISUAL_RESULT" \
        --mask-type visual \
        --mask-ratio 0.9 \
        --max-new-tokens 50
else
    echo "  Visual POPE结果已存在，跳过推理"
fi

# ==================== Step 4: 评估结果 ====================
echo ""
echo "[Step 4/4] 评估POPE结果..."

# 评估Audio POPE
if [ -f "$AUDIO_RESULT" ]; then
    echo "  评估Audio POPE..."
    python -m affect_r1.merbench.evaluate_pope \
        --input "$AUDIO_RESULT" \
        --output "$OUTPUT_DIR/pope_audio_evaluation.json"
fi

# 评估Visual POPE
if [ -f "$VISUAL_RESULT" ]; then
    echo "  评估Visual POPE..."
    python -m affect_r1.merbench.evaluate_pope \
        --input "$VISUAL_RESULT" \
        --output "$OUTPUT_DIR/pope_visual_evaluation.json"
fi

echo ""
echo "=========================================="
echo "POPE实验完成!"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "文件列表:"
ls -la "$OUTPUT_DIR"

