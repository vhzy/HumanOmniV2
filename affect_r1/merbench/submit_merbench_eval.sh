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
GPUS=1

if [[ "$PARTITION" == "err-nodes" || "$PARTITION" == "r1-m1" || "$PARTITION" == "r1-m1-large" ]]; then
    DEVICE="N6lS.Iq.I10"
else
    DEVICE="N6lS.Iu.I80"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKDIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/merbench_eval_${DATE}.log"

# Environment setup
ENV_COMMAND="source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /mnt/afs/hanzhiyuan/miniconda3/etc/profile.d/conda.sh && \
conda activate /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2"

MODEL_PATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_5k_3
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_5k_3/inference
RUN_NAME=merbench_baseline
CKPT_NAME=checkpoint-1958
DATASETS="CMUMOSI CMUMOSEI SIMS "

# Evaluation command
EXE_COMMAND="cd ${PROJECT_ROOT} && \
export PYTHONPATH=${REPO_ROOT}:\$PYTHONPATH && \
for ds in \$DATASETS; do \
  ds_lower=\"\${ds,,}\"; \
  RESULT_DIR=${OUTPUT_ROOT}/results-\${ds_lower}/${RUN_NAME}; \
  LOG_FILE=\${RESULT_DIR}/${CKPT_NAME}_eval.txt; \
  mkdir -p \${RESULT_DIR}; \
  CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.evaluation \
    --results-root \"\${RESULT_DIR}\" \
    --dataset \"\${ds}\" \
    --dataset-root ${DATASET_ROOT} \
    --checkpoint-name ${CKPT_NAME} \
    --llm-name Qwen25 \
    --log-file \"\${LOG_FILE}\"; \
done"

COMMAND="cd \"$WORKDIR\" && ${ENV_COMMAND} && ${EXE_COMMAND} >> \"${LOG_FILE}\" 2>&1"

echo ">>> MERBench eval job submitted"
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
--job-name "affect_r1_merbench_eval" \
--command "$COMMAND"

