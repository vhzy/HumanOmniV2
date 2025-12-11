#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# AffectGPT-R1: GRPO training entry that mirrors the official run_grpo_qwenomni
# ---------------------------------------------------------------------------

# --- Distributed defaults (can be overridden externally) ---
WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-16666}
RANK=${RANK:-0}

# --- Experiment metadata ---
RUN_NAME=${RUN_NAME:-affect_r1_grpo15}
OUTPUT_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_grpo.sh"
export LOG_PATH="${OUTPUT_DIR}/debug_log_${RUN_NAME}.txt"

# --- Environment preparation ---
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO

SFT_MODEL_PATH="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline7/checkpoint-1958"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_config.yaml"
QWEN_OMNI_UTILS="/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src"

export PYTHONPATH=${PYTHONPATH:-}:/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1:/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/src:$QWEN_OMNI_UTILS
export WANDB_PROJECT=${WANDB_PROJECT:-affect_r1_baseline_13}
# 强制在线同步到 wandb（用户提供的 API key）
export WANDB_API_KEY="${WANDB_API_KEY:-efaf785d107068d0e915c70c200861194ee42231}"
export WANDB_MODE=online

# --- Launch GRPO training (torchrun + Deepspeed) ---
torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/src/open_r1/grpo_qwenomni.py \
    --deepspeed /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero3_offload.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $SFT_MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --learning_rate 1e-5 \
    --beta 0.04 \
    --epsilon 0.2 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --scale_rewards false \
    --reward_funcs "affect_reward.emotion_wheel_reward" "affect_reward.format_reward"  \
    --reward_weights 1.0 0.1 \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model false \
    --dataloader_num_workers 16 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

exit ${PIPESTATUS[0]}
