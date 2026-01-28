#!/usr/bin/env bash

# 提交任务信息
WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
# PARTITION=amplarge2
PARTITION=m-train-ocr
# PARTITION=h100-share3
# PARTITION=m-train-1
# PARTITION=m-train-1
# PARTITION=m-train-ocr
# PARTITION=vqalarge2
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image
MOUNT=1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs

nodes=1
GPUS=2

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iq.I10"
else
    DEVICE="N6lS.Iu.I80"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$WORKDIR/HumanOmniV2/affect_r1/output/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sft_baseline_$DATE.log"

# 假设你已经配置好了环境，这里直接运行 run_sft.sh
# 注意：这里需要根据你的实际环境配置修改 conda env
# 如果不需要conda环境，可以直接去掉 source 和 conda activate
# ENV_COMMAND="echo 'Using default environment'"
ENV_COMMAND="source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2"

EXE_COMMAND="sh /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/run_sft.sh"
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
--worker-spec "${DEVICE}.${GPUS}" \
--worker-nodes "$nodes" \
--job-name "affect_r1_sft_baseline" \
--command "$COMMAND"

