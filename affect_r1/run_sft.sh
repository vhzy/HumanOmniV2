#!/bin/bash

# SFT Training Script for AffectGPT-R1 baseline (HumanOmniV2 framework)
# Model: Qwen2.5-Omni
# Data: MER2025 track3_train_mercaptionplus (reformatted)

# Ensure conda environment is activated
# source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
# conda activate humanomni
# Assume environment is already activated by submit script or manually

# Distributed training settings
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${8:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

RUN_NAME="affect_r1_sft"
MODEL_PATH="/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/sft_config.yaml"
OUTPUT_DIR="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline_face_1"
DEEPSPEED_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero2.json"

export LOG_PATH="$OUTPUT_DIR/debug_log_$RUN_NAME.txt"

# Enable debugging output
# export DEBUG_MODE="true"
# export LOG_PATH="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/debug/sft_debug.log"
# # Check if we need to relaunch with bash
# if [ -z "$BASH_VERSION" ]; then
#     exec bash "$0" "$@"
# fi

# Force activation of conda environment within the script to ensure it persists
# This handles cases where 'sh' execution might have lost the environment context
source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

mkdir -p $OUTPUT_DIR

cd /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal

# Add src directory to PYTHONPATH
export PYTHONPATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal:/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src:$PYTHONPATH

python -m torch.distributed.run --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/sft.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --freeze_vision_modules true \
    --use_audio_in_video false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 2 \
    --learning_rate 2e-5 \
    --bf16 true \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --save_strategy steps \
    --log_level info \
    --save_only_model true 2>&1 | tee $OUTPUT_DIR/train.log

