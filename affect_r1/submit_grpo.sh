#!/usr/bin/env bash

# 提交任务信息
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
PARTITION=amplarge2
# PARTITION=h100-share2
# PARTITION=h100-share3
# PARTITION=m-train-1
# PARTITION=m-train-2
# PARTITION=m-train-ocr
# PARTITION=vqalarge2
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs

nodes=2
GPUS=8

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iu.I80"
else
    DEVICE="N6lS.Iu.I80"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Training parameters
RUN_NAME="affect_r1_grpo_5k_11"
OUTPUT_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

SFT_MODEL_PATH="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline_5k_2/checkpoint-314"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_config.yaml"
DEEPSPEED_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero3_offload.json"

# 日志写入训练输出目录
mkdir -p "$OUTPUT_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/grpo_train_$DATE.log"

# Environment setup command
# Use the humanomni_v2 environment (same approach as submit_sft_fixed.sh)
ENV_COMMAND="source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && \
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2 && \
pip install openpyxl"
# Training command
# 多节点训练需要动态设置master_addr和node_rank
# sco acp会自动设置环境变量：MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
EXE_COMMAND="cd /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal && \
export PYTHONPATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1:/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/src:/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src:\$PYTHONPATH && \
export NCCL_SOCKET_TIMEOUT=3600 && \
export NCCL_DEBUG=INFO && \
export NCCL_IB_DISABLE=0 && \
export NCCL_SOCKET_IFNAME=eth0 && \
export WANDB_PROJECT=$RUN_NAME && \
export WANDB_API_KEY=efaf785d107068d0e915c70c200861194ee42231 && \
export WANDB_MODE=online && \
export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1} && \
export MASTER_PORT=\${MASTER_PORT:-16666} && \
export NODE_RANK=\${RANK:-0} && \
torchrun --nproc_per_node $GPUS --nnodes $nodes --node_rank \$NODE_RANK --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $SFT_MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --learning_rate 5e-6 \
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
    --reward_funcs affect_reward.emotion_wheel_reward affect_reward.format_reward \
    --reward_weights 1.0 0.1 \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model false \
    --dataloader_num_workers 16"

COMMAND="cd \"$WORKDIR\" && ${ENV_COMMAND} && ${EXE_COMMAND} >> \"${LOG_FILE}\" 2>&1"

echo ">>> GRPO Job submitted"
echo "Command: ${COMMAND}"
echo "Log file: ${LOG_FILE}"

sco acp jobs create \
--workspace-name "$WORKSPACE" \
-p "$PARTITION" \
--container-image-url "$CONTAINTER" \
--storage-mount "$MOUNT" \
--training-framework pytorch \
--worker-spec "${DEVICE}.${GPUS}" \
--worker-nodes "$nodes" \
--job-name "affect_r1_grpo_baseline" \
--command "$COMMAND"

