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

nodes=1
GPUS=8
CPU=64
MEM=1024

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iq.I10"
else
    DEVICE="N6lS.Iq.I10"
fi

#N6lS.Iq.I10.8.64c1024g
#N6lS.Iu.I80.1.4c64g

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Training parameters
RUN_NAME="affect_r1_sft"
MODEL_PATH="/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B"
DATA_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/sft_config.yaml"
OUTPUT_DIR="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline_5k_2"
DEEPSPEED_CONFIG="/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal/run_scripts/zero2.json"
NPROC_PER_NODE=8

# 日志写入训练输出目录
mkdir -p "$OUTPUT_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/sft_baseline_$DATE.log"

# Environment setup command
# Use the newly created humanomni_v2 environment (shared between dev machine and container)
ENV_COMMAND="source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && \
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2"

# Training command
EXE_COMMAND="cd /mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal && \
export PYTHONPATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal:/mnt/afs/hanzhiyuan/code/Qwen2.5-Omni/qwen-omni-utils/src:\$PYTHONPATH && \
python -m torch.distributed.run --nproc_per_node $GPUS --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=16667 \
    src/open_r1/sft.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --freeze_vision_modules true \
    --use_audio_in_video false \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
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
    --save_only_model true"

COMMAND="cd \"$WORKDIR\" && ${ENV_COMMAND} && ${EXE_COMMAND} >> \"${LOG_FILE}\" 2>&1"

echo ">>> SFT Job submitted"
echo "Command: ${COMMAND}"
echo "Log file: ${LOG_FILE}"

sco acp jobs create \
--workspace-name "$WORKSPACE" \
-p "$PARTITION" \
--container-image-url "$CONTAINTER" \
--storage-mount "$MOUNT" \
--training-framework pytorch \
--worker-spec "${DEVICE}.${GPUS}.${CPU}.${MEM}g" \
--worker-nodes "$nodes" \
--job-name "affect_r1_sft_baseline" \
--command "$COMMAND"

