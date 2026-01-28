#!/bin/bash
# HumanOmniV2 评测一键脚本
#
# 任务1: 生成描述 + 线索提取
# 任务2: POPE 压力测试
#
# 使用示例：
#
# 任务1 - 生成描述：
# CUDA_VISIBLE_DEVICES=0 bash run_humanomni_eval.sh description
#
# 任务1 - 线索提取（在描述生成后）：
# CUDA_VISIBLE_DEVICES=0 bash run_humanomni_eval.sh clue
#
# 任务2 - POPE 音频幻觉测试：
# CUDA_VISIBLE_DEVICES=0 bash run_humanomni_eval.sh pope_audio
#
# 任务2 - POPE 视觉幻觉测试：
# CUDA_VISIBLE_DEVICES=1 bash run_humanomni_eval.sh pope_visual
#
# 使用自定义模型路径：
# MODEL_PATH=/path/to/checkpoint bash run_humanomni_eval.sh description

set -e

# ==================== 配置 ====================
TASK=${1:-"description"}

# 模型路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-"/mnt/afs/hanzhiyuan/huggingface/humanomniv2"}
PROCESSOR_PATH=${PROCESSOR_PATH:-"/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B"}
DATASET_ROOT=${DATASET_ROOT:-"/mnt/afs/hanzhiyuan/datasets"}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/humanomniv2_baseline"}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# POPE 数据集路径
POPE_DATA_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset"

# 线索提取配置
ENGINE=${ENGINE:-"qwen"}
LLM_NAME=${LLM_NAME:-"Qwen25"}

echo "=============================================="
echo "HumanOmniV2 Evaluation Pipeline"
echo "=============================================="
echo "Task: $TASK"
echo "Model Path: $MODEL_PATH"
echo "Processor Path: $PROCESSOR_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

case $TASK in
    "description")
        # 任务1: 生成描述
        echo ""
        echo "[Task 1] Generating emotion descriptions..."
        python "${SCRIPT_DIR}/run_description_humanomni.py" \
            --model-path "$MODEL_PATH" \
            --processor-path "$PROCESSOR_PATH" \
            --dataset-root "$DATASET_ROOT" \
            --output "${OUTPUT_DIR}/ovmerd_description.jsonl" \
            --max-new-tokens 1024 \
            --temperature 0.9 \
            --top-p 0.9 \
            --do-sample \
            --use-audio-in-video \
            --resume
        ;;
    
    "clue")
        # 任务1: 线索提取
        echo ""
        echo "[Task 1] Extracting multimodal clues..."
        DESCRIPTION_FILE="${OUTPUT_DIR}/ovmerd_description.jsonl"
        if [ ! -f "$DESCRIPTION_FILE" ]; then
            echo "Error: Description file not found: $DESCRIPTION_FILE"
            echo "Please run 'description' task first."
            exit 1
        fi
        
        python "${SCRIPT_DIR}/check_hallucination.py" \
            --input "$DESCRIPTION_FILE" \
            --output "${OUTPUT_DIR}/clue_extraction_${ENGINE}.jsonl" \
            --engine "$ENGINE" \
            --llm-name "$LLM_NAME" \
            --dataset-root "$DATASET_ROOT"
        ;;
    
    "pope_audio")
        # 任务2: POPE 音频幻觉测试
        echo ""
        echo "[Task 2] POPE Audio Hallucination Test..."
        python "${SCRIPT_DIR}/run_pope_inference.py" \
            --model-path "$MODEL_PATH" \
            --processor-path "$PROCESSOR_PATH" \
            --pope-dataset "${POPE_DATA_ROOT}/pope_audio_filtered_v2.jsonl" \
            --output "${OUTPUT_DIR}/pope_audio_mask.jsonl" \
            --mask-type audio \
            --mask-ratio 1.0 \
            --max-new-tokens 50 \
            --resume
        ;;
    
    "pope_visual")
        # 任务2: POPE 视觉幻觉测试
        echo ""
        echo "[Task 2] POPE Visual Hallucination Test..."
        python "${SCRIPT_DIR}/run_pope_inference.py" \
            --model-path "$MODEL_PATH" \
            --processor-path "$PROCESSOR_PATH" \
            --pope-dataset "${POPE_DATA_ROOT}/pope_visual_filtered_v2.jsonl" \
            --output "${OUTPUT_DIR}/pope_visual_mask.jsonl" \
            --mask-type visual \
            --mask-ratio 1.0 \
            --max-new-tokens 50 \
            --resume
        ;;
    
    "all")
        # 运行所有任务
        echo ""
        echo "Running all tasks..."
        
        echo "[1/4] Generating descriptions..."
        bash "$0" description
        
        echo "[2/4] Extracting clues..."
        bash "$0" clue
        
        echo "[3/4] POPE audio test..."
        bash "$0" pope_audio
        
        echo "[4/4] POPE visual test..."
        bash "$0" pope_visual
        ;;
    
    *)
        echo "Unknown task: $TASK"
        echo "Usage: $0 {description|clue|pope_audio|pope_visual|all}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Task '$TASK' completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

