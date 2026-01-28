#!/bin/bash
set -euo pipefail

# AFFECTGPT_ROOT=/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo_logit2/inference-cf
RUN_NAME=merbench_cf_old_10_early_answer
CKPT_NAME=checkpoint-3000
DATASETS=( OVMERDPlus MER2023 MER2024 MELD IEMOCAPFour CMUMOSI CMUMOSEI SIMS SIMSV2 ) #OVMERDPlus MER2023 MER2024 MELD IEMOCAPFour CMUMOSI CMUMOSEI SIMS SIMSV2 MER2025

for ds in "${DATASETS[@]}"; do
  ds_lower="${ds,,}"
  RESULT_DIR="${OUTPUT_ROOT}/results-${ds_lower}/${RUN_NAME}"
  LOG_FILE="${RESULT_DIR}/${CKPT_NAME}_eval.txt"
  mkdir -p "${RESULT_DIR}"
  CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.evaluation \
    --results-root "${RESULT_DIR}" \
    --dataset "${ds}" \
    --dataset-root "${DATASET_ROOT}" \
    --checkpoint-name "${CKPT_NAME}" \
    --llm-name Qwen25 \
    --log-file "${LOG_FILE}"
done

# --no-use-answer-opensetbash