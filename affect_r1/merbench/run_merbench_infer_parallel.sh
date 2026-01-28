#!/bin/bash
# =============================================================================
# 本地并行推理脚本：指定GPU同时跑多个数据集
# 用法: ./run_merbench_infer_parallel.sh
# =============================================================================

# GPU配置：指定要使用的GPU编号（用空格分隔）
GPU_IDS=( 0 1 )
NUM_GPUS=${#GPU_IDS[@]}

# 路径配置
MODEL_PATH=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline7/checkpoint-1958
PROCESSOR_PATH=$MODEL_PATH
DATASET_ROOT=/mnt/afs/hanzhiyuan/datasets
OUTPUT_ROOT=/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/sft_baseline7/checkpoint-1958/inference2
RUN_NAME=merbench_baseline
CKPT_NAME=checkpoint-3262

# 数据集（用逗号分隔）
DATASETS="OVMERDPlus,MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2" #,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2

# 目录设置
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKDIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo ">>> 本地并行模式：使用 ${NUM_GPUS} 张GPU (GPU: ${GPU_IDS[*]})"
echo ">>> 模型路径: ${MODEL_PATH}"
echo ">>> 输出目录: ${OUTPUT_ROOT}"
echo ">>> 日志目录: ${LOG_DIR}"
echo "=============================================="

# 延迟启动6小时
# echo ">>> 将在 6 小时后启动所有推理任务..."
# sleep 6h

# 将数据集字符串转为数组
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
TOTAL_DATASETS=${#DATASET_ARRAY[@]}
echo ">>> 共有 ${TOTAL_DATASETS} 个数据集: ${DATASETS}"
echo ""

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# 存储后台进程PID和对应的数据集
declare -a PIDS
declare -a GPU_DATASETS

# 第一批：在指定GPU上并行启动数据集
echo ">>> [阶段1] 启动数据集..."
echo ""

for i in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $i -lt $TOTAL_DATASETS ]; then
        DATASET="${DATASET_ARRAY[$i]}"
        GPU_ID=${GPU_IDS[$i]}
        TASK_LOG="${LOG_DIR}/merbench_infer_${DATASET}_gpu${GPU_ID}_${DATE}.log"
        
        echo ">>> [GPU ${GPU_ID}] 启动数据集: ${DATASET}"
        echo "    日志文件: ${TASK_LOG}"
        
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -m affect_r1.merbench.run_inference \
            --model-path "${MODEL_PATH}" \
            --processor-path "${PROCESSOR_PATH}" \
            --dataset-root "${DATASET_ROOT}" \
            --output-root "${OUTPUT_ROOT}" \
            --run-name "${RUN_NAME}" \
            --checkpoint-name "${CKPT_NAME}" \
            --datasets "${DATASET}" \
            --max-new-tokens 1024 \
            --temperature 0.9 \
            --top-p 0.9 \
            --do-sample \
            --use-audio-in-video > "${TASK_LOG}" 2>&1 &
        
        PIDS[$i]=$!
        GPU_DATASETS[$i]="${DATASET}"
        echo "    PID: ${PIDS[$i]}"
    fi
done

echo ""
echo ">>> 前 ${NUM_GPUS} 个数据集已启动"

# 处理剩余的数据集（第9个及之后）
NEXT_DATASET_IDX=$NUM_GPUS

while [ $NEXT_DATASET_IDX -lt $TOTAL_DATASETS ]; do
    echo ""
    echo ">>> [阶段2] 等待空闲GPU以启动剩余数据集..."
    
    while true; do
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            if [ -n "${PIDS[$i]}" ]; then
                # 检查进程是否还在运行
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    # 进程已完成
                    wait "${PIDS[$i]}"
                    EXIT_CODE=$?
                    echo ""
                    echo ">>> [GPU ${GPU_IDS[$i]}] 数据集 ${GPU_DATASETS[$i]} 完成，退出码: ${EXIT_CODE}"
                    
                    # 在该GPU上启动下一个数据集
                    if [ $NEXT_DATASET_IDX -lt $TOTAL_DATASETS ]; then
                        DATASET="${DATASET_ARRAY[$NEXT_DATASET_IDX]}"
                        GPU_ID=${GPU_IDS[$i]}
                        TASK_LOG="${LOG_DIR}/merbench_infer_${DATASET}_gpu${GPU_ID}_${DATE}.log"
                        
                        echo ">>> [GPU ${GPU_ID}] 启动数据集: ${DATASET}"
                        echo "    日志文件: ${TASK_LOG}"
                        
                        CUDA_VISIBLE_DEVICES=${GPU_ID} python -m affect_r1.merbench.run_inference \
                            --model-path "${MODEL_PATH}" \
                            --processor-path "${PROCESSOR_PATH}" \
                            --dataset-root "${DATASET_ROOT}" \
                            --output-root "${OUTPUT_ROOT}" \
                            --run-name "${RUN_NAME}" \
                            --checkpoint-name "${CKPT_NAME}" \
                            --datasets "${DATASET}" \
                            --max-new-tokens 1024 \
                            --temperature 0.9 \
                            --top-p 0.9 \
                            --do-sample \
                            --use-audio-in-video > "${TASK_LOG}" 2>&1 &
                        
                        PIDS[$i]=$!
                        GPU_DATASETS[$i]="${DATASET}"
                        echo "    PID: ${PIDS[$i]}"
                        NEXT_DATASET_IDX=$((NEXT_DATASET_IDX + 1))
                    fi
                    break 2
                fi
            fi
        done
        sleep 5  # 每5秒检查一次
    done
done

# 等待所有剩余进程完成
echo ""
echo ">>> [阶段3] 等待所有任务完成..."

for i in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -n "${PIDS[$i]}" ]; then
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            wait "${PIDS[$i]}"
            EXIT_CODE=$?
            echo ">>> [GPU ${GPU_IDS[$i]}] 数据集 ${GPU_DATASETS[$i]} 完成，退出码: ${EXIT_CODE}"
        fi
    fi
done

echo ""
echo "=============================================="
echo ">>> 所有 ${TOTAL_DATASETS} 个数据集推理完成！"
echo ">>> 日志目录: ${LOG_DIR}"
echo "=============================================="
