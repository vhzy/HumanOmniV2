#!/bin/bash
# set -euo pipefail

MODEL_PATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_5k_3
PROCESSOR_PATH=$MODEL_PATH       # 若 processor 即模型，可复用
# AFFECTGPT_ROOT=/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_5k_3/inference
RUN_NAME=merbench_baseline
CKPT_NAME=checkpoint-1958
DATASETS="SIMSV2"
# DATASETS="OVMERDPlus "OVMERDPlus, MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,

CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.run_inference \
  --model-path "${MODEL_PATH}" \
  --processor-path "${PROCESSOR_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --checkpoint-name "${CKPT_NAME}" \
  --datasets "${DATASETS}" \
  --max-new-tokens 1024 \
  --temperature 0.9 \
  --top-p 0.9 \
  --do-sample \
  --use-audio-in-video 