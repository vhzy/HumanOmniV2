#!/bin/bash
# Prompt Mode 对照实验脚本
# 测试显式询问被mask模态对模型幻觉的影响

set -e

# ==================== 配置区 ====================

# 模型和数据路径
MODEL_PATH="${MODEL_PATH:-/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_stage2_13}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/afs/hanzhiyuan/data/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

# 实验配置
RUN_BASE_NAME="${RUN_BASE_NAME:-prompt_mode_exp}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-3000}"
DATASETS="${DATASETS:-OVMERDPlus}"

# 推理参数
MAX_NEW_TOKENS=512
TEMPERATURE=0.9
TOP_P=0.9
MASK_RATIO=0.9

# ==================== 实验列表 ====================

# 定义实验配置: "mask_type:prompt_mode:experiment_name"
EXPERIMENTS=(
    # 组1: mask视觉 - 对比默认prompt vs 询问视觉
    "visual:default:mask_visual_default"
    "visual:ask_masked:mask_visual_ask"
    
    # 组2: mask听觉 - 对比默认prompt vs 询问听觉
    "audio:default:mask_audio_default"
    "audio:ask_masked:mask_audio_ask"
)

# ==================== 函数定义 ====================

run_inference() {
    local mask_modality=$1
    local prompt_mode=$2
    local exp_name=$3
    
    echo ""
    echo "=========================================="
    echo "实验: $exp_name"
    echo "  Mask模态: $mask_modality"
    echo "  Prompt模式: $prompt_mode"
    echo "=========================================="
    
    python run_inference_counterfactual.py \
        --model-path "$MODEL_PATH" \
        --dataset-root "$DATASET_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --run-name "${RUN_BASE_NAME}_${exp_name}" \
        --checkpoint-name "$CHECKPOINT_NAME" \
        --datasets "$DATASETS" \
        --max-new-tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --use-audio-in-video \
        --mask-modality "$mask_modality" \
        --mask-ratio $MASK_RATIO \
        --prompt-mode "$prompt_mode"
    
    if [ $? -eq 0 ]; then
        echo "✅ 实验 $exp_name 完成"
    else
        echo "❌ 实验 $exp_name 失败"
        return 1
    fi
}

# ==================== 主流程 ====================

echo "======================================"
echo "Prompt Mode 对照实验"
echo "======================================"
echo "模型路径: $MODEL_PATH"
echo "数据集根目录: $DATASET_ROOT"
echo "输出目录: $OUTPUT_ROOT"
echo "检查点: $CHECKPOINT_NAME"
echo "评估数据集: $DATASETS"
echo ""
echo "将运行 ${#EXPERIMENTS[@]} 个实验"
echo "======================================"

# 记录开始时间
START_TIME=$(date +%s)

# 运行所有实验
SUCCESS_COUNT=0
FAIL_COUNT=0

for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r mask_modality prompt_mode exp_name <<< "$exp_config"
    
    if run_inference "$mask_modality" "$prompt_mode" "$exp_name"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        echo "⚠️  继续执行剩余实验..."
    fi
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 输出总结
echo ""
echo "======================================"
echo "实验完成总结"
echo "======================================"
echo "总实验数: ${#EXPERIMENTS[@]}"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo "总耗时: ${DURATION}秒 (约 $((DURATION / 60))分钟)"
echo ""

# 显示结果文件位置
echo "结果文件位置:"
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r mask_modality prompt_mode exp_name <<< "$exp_config"
    RESULT_DIR="$OUTPUT_ROOT/${RUN_BASE_NAME}_${exp_name}/inference-cf/results-${DATASETS,,}"
    if [ -d "$RESULT_DIR" ]; then
        echo "  - $RESULT_DIR/"
    fi
done

echo ""
echo "======================================"
echo "下一步: 运行CHAIR-M评估"
echo "======================================"
echo ""
echo "# 评估 mask_visual_default"
echo "python chair_m_evaluation.py \\"
echo "  --jsonl-path $OUTPUT_ROOT/${RUN_BASE_NAME}_mask_visual_default/inference-cf/results-${DATASETS,,}/$CHECKPOINT_NAME.jsonl \\"
echo "  --output-csv mask_visual_default_scores.csv \\"
echo "  --mask-type mask_visual"
echo ""
echo "# 评估 mask_visual_ask"
echo "python chair_m_evaluation.py \\"
echo "  --jsonl-path $OUTPUT_ROOT/${RUN_BASE_NAME}_mask_visual_ask/inference-cf/results-${DATASETS,,}/$CHECKPOINT_NAME.jsonl \\"
echo "  --output-csv mask_visual_ask_scores.csv \\"
echo "  --mask-type mask_visual"
echo ""
echo "# 对比结果"
echo "python compare_chair_scores.py \\"
echo "  mask_visual_default_scores.csv mask_visual_ask_scores.csv \\"
echo "  --labels 'Visual-Default' 'Visual-AskMasked' \\"
echo "  --output-dir comparison_visual"
echo ""
echo "(同样的步骤适用于 mask_audio 实验)"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有实验成功完成！"
    exit 0
else
    echo "⚠️  部分实验失败，请检查日志"
    exit 1
fi

