#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# AffectGPT-R1: GRPO training with PAPO (Perception-Aware Policy Optimization)
# Multi-node cluster submission script
# ---------------------------------------------------------------------------

# 提交任务信息
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
# PARTITION=interleave
# PARTITION=h100-share3
PARTITION=amplarge2-large
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs

nodes=2
GPUS=8
CPU=64
MEM=1024

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iu.I80"
else
    DEVICE="N6lS.Iq.I10"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- PAPO Configuration ---
# PAPO Versions:
#   v0: Mask both audio and video simultaneously (simplest, 1 extra forward pass)
#   v1: Mask audio and video separately (2 extra forward passes)
#   v2: Matrix-guided fine-grained routing (2 extra forward passes, uses sim matrices)
PAPO_ENABLED=${PAPO_ENABLED:-true}
PAPO_VERSION=${PAPO_VERSION:-v2}  # v0, v1, or v2
PAPO_MASK_RATIO=${PAPO_MASK_RATIO:-0.7}      # Original PAPO: 60% blackening probability
PAPO_KL_COEF=${PAPO_KL_COEF:-8e-3}          # Original PAPO: 1e-3
PAPO_ENTROPY_COEF=${PAPO_ENTROPY_COEF:-3e-4} # Original PAPO: 0.03 (Set to 0.0 in env to disable)
PAPO_USE_NOISE=${PAPO_USE_NOISE:-false}      # Original PAPO: uses zeros (black), not noise
PAPO_ROUTING_THRESHOLD=${PAPO_ROUTING_THRESHOLD:-0.45}

# Training parameters
RUN_NAME="affect_r1_grpo2_papo_54_${PAPO_VERSION}"
OUTPUT_ROOT="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

SFT_MODEL_PATH="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline_5k_2/checkpoint-314"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_config.yaml"
DEEPSPEED_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero3_offload.json"

# 日志写入训练输出目录
mkdir -p "$OUTPUT_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/grpo_train_$DATE.log"

echo "========================================"
echo "PAPO Configuration:"
echo "  Enabled: ${PAPO_ENABLED}"
echo "  Version: ${PAPO_VERSION}"
echo "  Mask Ratio: ${PAPO_MASK_RATIO}"
echo "  KL Coef: ${PAPO_KL_COEF}"
echo "  Entropy Coef: ${PAPO_ENTROPY_COEF}"
echo "  Use Noise: ${PAPO_USE_NOISE}"
echo "  Routing Threshold: ${PAPO_ROUTING_THRESHOLD} (V2 only)"
echo "========================================"

# Environment setup command
ENV_COMMAND="source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && \
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2 && \
pip install openpyxl"

# Training commandnokill_告知_
# 多节点训练需要动态设置master_addr和node_rank
# sco acp会自动设置环境变量：MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
EXE_COMMAND="cd /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal && \
export PYTHONPATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1:/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/src:/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src:\$PYTHONPATH && \
export NCCL_SOCKET_TIMEOUT=5400 && \
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000 && \
export NCCL_DEBUG=INFO && \
export NCCL_IB_DISABLE=0 && \
export NCCL_SOCKET_IFNAME=eth0 && \
export WANDB_PROJECT=affect_r1_papo && \
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
    --learning_rate 2e-6 \
    --beta 0.04 \
    --epsilon 0.2 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --scale_rewards false \
    --reward_funcs affect_reward.emotion_wheel_reward affect_reward.format_reward affect_reward.rubric_perc_reward_with_matrices affect_reward.rubric_coh_reward \
    --reward_weights 1.0 0.1 0.5 0.5  \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model false \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 16 \
    --papo_enabled $PAPO_ENABLED \
    --papo_version $PAPO_VERSION \
    --papo_mask_ratio $PAPO_MASK_RATIO \
    --papo_kl_coef $PAPO_KL_COEF \
    --papo_entropy_coef $PAPO_ENTROPY_COEF \
    --papo_use_noise $PAPO_USE_NOISE \
    --papo_routing_threshold $PAPO_ROUTING_THRESHOLD"

COMMAND="cd \"$WORKDIR\" && ${ENV_COMMAND} && ${EXE_COMMAND} >> \"${LOG_FILE}\" 2>&1"

echo ">>> GRPO PAPO Job submitted"
echo "Command: ${COMMAND}"
echo "Log file: ${LOG_FILE}"

sco acp jobs create \
--workspace-name "$WORKSPACE" \
-p "$PARTITION" \
--container-image-url "$CONTAINTER" \
--storage-mount "$MOUNT" \
--training-framework pytorch \
--worker-spec "N6lS.Iu.I80.8.64c1024g" \
--worker-nodes "$nodes" \
--job-name "nokill_affect_r1_grpo_papo" \
--command "$COMMAND"

