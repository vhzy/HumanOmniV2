#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# AffectGPT-R1: GRPO Stage3 training - Local 8-GPU run
# ---------------------------------------------------------------------------

# --- Distributed defaults (can be overridden externally) ---
WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-16666}
RANK=${RANK:-0}

# --- Experiment metadata ---
RUN_NAME=${RUN_NAME:-affect_r1_grpo_stage3_8}
OUTPUT_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_grpo_stage3.sh"
export LOG_PATH="${OUTPUT_DIR}/grpo_train_$(date +%Y%m%d_%H%M%S).log"

# --- Training parameters ---
SFT_MODEL_PATH="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_stage2_5"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_config_stage3.yaml"
DEEPSPEED_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero3_offload.json"
QWEN_OMNI_UTILS="/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src"

# --- Environment preparation ---
export PYTHONPATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1:/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/src:$QWEN_OMNI_UTILS${PYTHONPATH:+:$PYTHONPATH}
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export WANDB_PROJECT=${WANDB_PROJECT:-$RUN_NAME}
export WANDB_API_KEY="${WANDB_API_KEY:-efaf785d107068d0e915c70c200861194ee42231}"
export WANDB_MODE=online

# --- Optional: Activate conda environment if needed ---
# Uncomment if you need to activate conda environment
# source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh
# conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2

# --- Launch GRPO training (torchrun + Deepspeed) ---
# echo ">>> GRPO Stage3 training will start after 3 hours"
# echo ">>> Sleeping for 3 hours before training..."
# sleep 3h
# echo ">>> Now starting GRPO Stage3 training"

cd /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $SFT_MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --learning_rate 2e-6 \
    --beta 0.04 \
    --epsilon 0.2 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --scale_rewards false \
    --reward_funcs affect_reward.emotion_wheel_reward affect_reward.format_reward logit_reward.coherence \
    --logit_reward_scale_method tanh \
    --reward_weights 1.0 0.1 0.5 \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --logit_reward_use_neg_contrast false \
    --save_steps 500 \
    --save_only_model false \
    --max_grad_norm 1.5 \
    --dataloader_num_workers 16 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

exit ${PIPESTATUS[0]}
