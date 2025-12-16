#!/bin/bash
set -euo pipefail

# AFFECTGPT_ROOT=/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_stage2_5/inference
RUN_NAME=merbench_baseline
CKPT_NAME=checkpoint-3262
DATASETS=(    IEMOCAPFour CMUMOSI CMUMOSEI SIMS SIMSV2) #OVMERDPlus MER2023 MER2024 MELD IEMOCAPFour CMUMOSI CMUMOSEI SIMS SIMSV2 MER2025

for ds in "${DATASETS[@]}"; do
  ds_lower="${ds,,}"
  RESULT_DIR="${OUTPUT_ROOT}/results-${ds_lower}/${RUN_NAME}"
  LOG_FILE="${RESULT_DIR}/${CKPT_NAME}_eval.txt"
  mkdir -p "${RESULT_DIR}"
  CUDA_VISIBLE_DEVICES=1 python -m affect_r1.merbench.evaluation \
    --results-root "${RESULT_DIR}" \
    --dataset "${ds}" \
    --dataset-root "${DATASET_ROOT}" \
    --checkpoint-name "${CKPT_NAME}" \
    --llm-name Qwen25 \
    --log-file "${LOG_FILE}"
done

# --no-use-answer-opensetbash