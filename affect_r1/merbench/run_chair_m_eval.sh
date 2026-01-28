#!/bin/bash
# Batch evaluation script for CHAIR-M metric

set -euo pipefail

# Configuration
API_KEY="${OPENAI_API_KEY:-your_api_key_here}"
MODEL="gpt-4o-mini"  # or "gpt-4o" for better quality
# BASE_URL=""  # Optional: for using local models

# Base paths
BASE_OUTPUT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output
CHECKPOINT="checkpoint-3000"

# Experiments to evaluate
EXPERIMENTS=(
    "affect_r1_grpo_stage2_13:merbench_cf10_mask_audio:mask_audio"
    "affect_r1_grpo_stage2_13:merbench_cf10_mask_visual:mask_visual"
    # Add more experiments here in format "model_dir:run_name:mask_type"
)

# Datasets
DATASETS=(
    "OVMERDPlus"
    # "MER2023"
    # "MER2024"
    # Add more datasets as needed
)

echo "=========================================="
echo "CHAIR-M Batch Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "=========================================="
echo ""

# Process each experiment
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r model_dir run_name mask_type <<< "$exp"
    
    echo "Processing: $model_dir / $run_name ($mask_type)"
    
    for dataset in "${DATASETS[@]}"; do
        dataset_lower="${dataset,,}"
        
        # Construct paths
        JSONL_PATH="${BASE_OUTPUT}/${model_dir}/inference-cf/${run_name}/${CHECKPOINT}/${dataset_lower}/results.jsonl"
        
        # Alternative path structure (if results are directly in run_name folder)
        if [ ! -f "$JSONL_PATH" ]; then
            JSONL_PATH="${BASE_OUTPUT}/${model_dir}/inference-cf/results-${dataset_lower}/${run_name}/${CHECKPOINT}.jsonl"
        fi
        
        # Check if file exists
        if [ ! -f "$JSONL_PATH" ]; then
            echo "  [SKIP] $dataset - File not found: $JSONL_PATH"
            continue
        fi
        
        # Output CSV path
        OUTPUT_DIR="$(dirname "$JSONL_PATH")"
        OUTPUT_CSV="${OUTPUT_DIR}/${CHECKPOINT}_chair_m_scores.csv"
        
        echo "  [EVAL] $dataset"
        echo "    Input: $JSONL_PATH"
        echo "    Output: $OUTPUT_CSV"
        
        # Run evaluation
        python -m affect_r1.merbench.chair_m_evaluation \
            --jsonl-path "$JSONL_PATH" \
            --output-csv "$OUTPUT_CSV" \
            --api-key "$API_KEY" \
            --model "$MODEL" \
            --mask-type "$mask_type" \
            2>&1 | tee "${OUTPUT_DIR}/${CHECKPOINT}_chair_m_eval.log"
        
        echo "    Done!"
        echo ""
    done
    
    echo "Finished experiment: $run_name"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

