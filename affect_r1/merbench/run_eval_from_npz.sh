#!/bin/bash
set -euo pipefail

AFFECT_OUTPUT=${AFFECT_OUTPUT:-/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output}
RUN_NAME=${RUN_NAME:-emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100}
CKPT_NAME=${CKPT_NAME:-checkpoint_000030_loss_0.602}
DATASET_ROOT=${DATASET_ROOT:-/mnt/afs/hanzhiyuan/datasets}

DATASETS=(MER2023 MER2024 MELD IEMOCAPFour CMUMOSI CMUMOSEI SIMS SIMSV2 OVMERDPlus)

echo "AFFECT_OUTPUT=${AFFECT_OUTPUT}"
echo "RUN_NAME=${RUN_NAME}"
echo "CKPT_NAME=${CKPT_NAME}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "Datasets: ${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  ds_lower="${ds,,}"
  base_dir="${AFFECT_OUTPUT}/results-${ds_lower}/${RUN_NAME}"
  checkpoint_base="${base_dir}/${CKPT_NAME}"
  if [[ ! -f "${checkpoint_base}.npz" ]]; then
    echo "[WARN] Missing ${checkpoint_base}.npz for ${ds}, skip."
    continue
  fi
  echo ">>> Evaluating ${ds}"
  python -m affect_r1.merbench.eval_from_npz \
    --dataset "${ds}" \
    --dataset-root "${DATASET_ROOT}" \
    --checkpoint-base "${checkpoint_base}"
done

