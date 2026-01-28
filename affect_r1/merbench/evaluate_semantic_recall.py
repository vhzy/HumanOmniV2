#!/usr/bin/env python3
"""
语义感知质量评估脚本 (Semantic Perception Evaluation)

功能：
1. 加载GT和Pred的线索提取结果
2. 使用Qwen3-embedding-0.6B计算语义相似度
3. 计算视觉/音频线索的召回率和精确率
4. 保存结果到模型输出目录

使用示例：
# 默认使用加权方式（推荐）
CUDA_VISIBLE_DEVICES=1 python -m affect_r1.merbench.evaluate_semantic_recall \
    --gt-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl \
    --pred-clues  /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_13/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/clue_extraction_Qwen.jsonl \
    --threshold 0.6

CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.evaluate_semantic_recall \
    --gt-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl \
    --pred-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo44_v0/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/clue_extraction_Qwen.jsonl\
    --threshold 0.6

# 使用合并embedding方式（允许跨模态匹配）
CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.evaluate_semantic_recall \
    --gt-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl \
    --pred-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/inference_cf/results-ovmerdplus/merbench_cf_new_10_baseline/hallucination_results_Qwen.jsonl \
    --threshold 0.5 \
    --overall-method combined
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm


def load_embedder():
    """加载Qwen3-embedding-0.6B模型"""
    from transformers import AutoTokenizer, AutoModel
    
    model_path = "/mnt/afs/hanzhiyuan/huggingface/Qwen3-Embedding-0.6B"
    
    print(f"[INFO] 加载embedding模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    return tokenizer, model


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """使用最后一个token的隐藏状态作为句子表示"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def encode_texts(texts: List[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    """批量编码文本为embedding向量"""
    if not texts:
        return np.array([])
    
    # 过滤空文本，保留索引映射
    valid_texts = []
    valid_indices = []
    for i, t in enumerate(texts):
        if t and isinstance(t, str) and len(t.strip()) > 0:
            valid_texts.append(t.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        return np.array([])
    
    all_embeddings = []
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # 检查是否有 NaN
            if torch.isnan(embeddings).any():
                print(f"    [WARNING] Found NaN in embeddings before normalization at batch {i}")
            
            # L2归一化（处理零向量情况）
            norms = embeddings.norm(dim=1, keepdim=True)
            # 将零范数替换为1以避免除以0
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            embeddings = embeddings / norms
            
            # 再次检查 NaN
            if torch.isnan(embeddings).any():
                print(f"    [WARNING] Found NaN in embeddings after normalization at batch {i}")
                # 将 NaN 替换为 0
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    result = np.concatenate(all_embeddings, axis=0)
    
    # 最终检查
    nan_count = np.isnan(result).sum()
    if nan_count > 0:
        print(f"    [WARNING] Final embeddings contain {nan_count} NaN values, replacing with 0")
        result = np.nan_to_num(result, nan=0.0)
    
    return result


def compute_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """计算两组embedding之间的余弦相似度矩阵"""
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return np.array([])
    
    # 由于已经L2归一化，点积即为余弦相似度
    return np.dot(embeddings_a, embeddings_b.T)


def compute_sample_metrics(
    gt_cues: List[str],
    pred_cues: List[str],
    gt_emb: np.ndarray,
    pred_emb: np.ndarray,
    threshold: float = 0.7
) -> Dict:
    """
    计算单个样本的召回率和精确率
    
    返回: {
        'recall': float,
        'precision': float,
        'matched_gt': int,
        'total_gt': int,
        'matched_pred': int,
        'total_pred': int,
        'valid_recall': bool,  # 召回率是否有效（GT不为空）
        'valid_precision': bool,  # 精确率是否有效（Pred不为空）
    }
    """
    total_gt = len(gt_cues)
    total_pred = len(pred_cues)
    
    # 处理边界情况
    if total_gt == 0 and total_pred == 0:
        # 都为空，视为完美匹配
        return {
            'recall': 1.0, 'precision': 1.0,
            'matched_gt': 0, 'total_gt': 0,
            'matched_pred': 0, 'total_pred': 0,
            'valid_recall': False, 'valid_precision': False
        }
    
    if total_gt == 0:
        # GT为空，没有需要召回的
        return {
            'recall': 1.0, 'precision': 0.0,  # 所有Pred都是多余的
            'matched_gt': 0, 'total_gt': 0,
            'matched_pred': 0, 'total_pred': total_pred,
            'valid_recall': False, 'valid_precision': True
        }
    
    if total_pred == 0:
        # Pred为空，没有召回任何GT
        return {
            'recall': 0.0, 'precision': 1.0,  # 没有错误的Pred
            'matched_gt': 0, 'total_gt': total_gt,
            'matched_pred': 0, 'total_pred': 0,
            'valid_recall': True, 'valid_precision': False
        }
    
    # 正常情况：都不为空
    # 检查embedding是否有效
    if len(gt_emb) == 0 or len(pred_emb) == 0:
        return {
            'recall': 0.0, 'precision': 0.0,
            'matched_gt': 0, 'total_gt': total_gt,
            'matched_pred': 0, 'total_pred': total_pred,
            'valid_recall': True, 'valid_precision': True
        }
    
    # 计算相似度矩阵 [gt_size, pred_size]
    sim_matrix = np.dot(gt_emb, pred_emb.T)
    
    # 召回率：对于每个GT，找Pred中最相似的
    max_sim_gt = sim_matrix.max(axis=1)  # [gt_size]
    matched_gt = int((max_sim_gt >= threshold).sum())
    recall = matched_gt / total_gt
    
    # 精确率：对于每个Pred，找GT中最相似的
    max_sim_pred = sim_matrix.max(axis=0)  # [pred_size]
    matched_pred = int((max_sim_pred >= threshold).sum())
    precision = matched_pred / total_pred
    
    return {
        'recall': recall, 'precision': precision,
        'matched_gt': matched_gt, 'total_gt': total_gt,
        'matched_pred': matched_pred, 'total_pred': total_pred,
        'valid_recall': True, 'valid_precision': True
    }


def compute_recall_precision(
    gt_claims: List[str],
    pred_claims: List[str],
    gt_embeddings: np.ndarray,
    pred_embeddings: np.ndarray,
    threshold: float = 0.7,
    debug: bool = False,
    return_details: bool = False
) -> Tuple:
    """
    计算召回率和精确率
    
    返回: 
    - 如果 return_details=False: (recall, precision, matched_gt, total_gt, matched_pred, total_pred)
    - 如果 return_details=True: (recall, precision, matched_gt, total_gt, matched_pred, total_pred, 
                                  unmatched_gt_indices, unmatched_pred_indices, 
                                  max_sim_gt, max_sim_pred)
    """
    total_gt = len(gt_claims)
    total_pred = len(pred_claims)
    
    if debug:
        print(f"    [DEBUG] total_gt={total_gt}, total_pred={total_pred}")
        print(f"    [DEBUG] gt_embeddings.shape={gt_embeddings.shape if hasattr(gt_embeddings, 'shape') else 'N/A'}")
        print(f"    [DEBUG] pred_embeddings.shape={pred_embeddings.shape if hasattr(pred_embeddings, 'shape') else 'N/A'}")
    
    if total_gt == 0:
        if return_details:
            return 0.0, 0.0, 0, 0, 0, total_pred, [], list(range(total_pred)), np.array([]), np.array([])
        return 0.0, 0.0, 0, 0, 0, total_pred
    
    if total_pred == 0:
        if return_details:
            return 0.0, 0.0, 0, total_gt, 0, 0, list(range(total_gt)), [], np.array([]), np.array([])
        return 0.0, 0.0, 0, total_gt, 0, 0
    
    # 检查embedding数组是否为空
    if len(gt_embeddings) == 0 or len(pred_embeddings) == 0:
        print(f"    [WARNING] Empty embeddings! gt_emb len={len(gt_embeddings)}, pred_emb len={len(pred_embeddings)}")
        if return_details:
            return 0.0, 0.0, 0, total_gt, 0, total_pred, list(range(total_gt)), list(range(total_pred)), np.array([]), np.array([])
        return 0.0, 0.0, 0, total_gt, 0, total_pred
    
    # 计算相似度矩阵 [gt_size, pred_size]
    sim_matrix = compute_similarity_matrix(gt_embeddings, pred_embeddings)
    
    if debug:
        print(f"    [DEBUG] sim_matrix.shape={sim_matrix.shape if hasattr(sim_matrix, 'shape') else 'N/A'}")
        if len(sim_matrix) > 0:
            print(f"    [DEBUG] sim_matrix min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}, mean={sim_matrix.mean():.4f}")
    
    # 检查相似度矩阵是否有效
    if sim_matrix.size == 0:
        print(f"    [WARNING] Empty similarity matrix!")
        if return_details:
            return 0.0, 0.0, 0, total_gt, 0, total_pred, list(range(total_gt)), list(range(total_pred)), np.array([]), np.array([])
        return 0.0, 0.0, 0, total_gt, 0, total_pred
    
    # 召回率：对于每个GT线索，找Pred中最相似的
    max_sim_gt = sim_matrix.max(axis=1)  # [gt_size]
    matched_gt = int((max_sim_gt >= threshold).sum())
    recall = matched_gt / total_gt
    unmatched_gt_indices = np.where(max_sim_gt < threshold)[0].tolist()
    
    if debug:
        print(f"    [DEBUG] max_sim_gt: min={max_sim_gt.min():.4f}, max={max_sim_gt.max():.4f}, mean={max_sim_gt.mean():.4f}")
    
    # 精确率：对于每个Pred线索，找GT中最相似的
    max_sim_pred = sim_matrix.max(axis=0)  # [pred_size]
    matched_pred = int((max_sim_pred >= threshold).sum())
    precision = matched_pred / total_pred
    unmatched_pred_indices = np.where(max_sim_pred < threshold)[0].tolist()
    
    if debug:
        print(f"    [DEBUG] max_sim_pred: min={max_sim_pred.min():.4f}, max={max_sim_pred.max():.4f}, mean={max_sim_pred.mean():.4f}")
    
    if return_details:
        return (recall, precision, matched_gt, total_gt, matched_pred, total_pred, 
                unmatched_gt_indices, unmatched_pred_indices, max_sim_gt, max_sim_pred)
    
    return recall, precision, matched_gt, total_gt, matched_pred, total_pred


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="语义感知质量评估")
    parser.add_argument("--gt-clues", type=str, required=True,
                        help="GT线索文件路径 (JSONL格式)")
    parser.add_argument("--pred-clues", type=str, required=True,
                        help="Pred线索文件路径 (hallucination_results_*.jsonl)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="相似度阈值 (默认0.7)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认为pred文件所在目录)")
    parser.add_argument("--save-details", action="store_true",
                        help="保存未匹配线索的详细信息")
    parser.add_argument("--overall-method", type=str, default="weighted",
                        choices=["weighted", "combined"],
                        help="整体指标计算方法：weighted=分别计算后加权(默认), combined=合并embedding后计算")
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.pred_clues).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("语义感知质量评估 (Semantic Perception Evaluation)")
    print("=" * 60)
    print(f"GT文件: {args.gt_clues}")
    print(f"Pred文件: {args.pred_clues}")
    print(f"相似度阈值: {args.threshold}")
    print(f"整体指标方法: {args.overall_method}")
    if args.overall_method == "weighted":
        print("  -> 分别计算视觉和音频指标后加权（严格区分模态）")
    else:
        print("  -> 合并embedding后计算（允许跨模态匹配）")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 加载数据
    print("[1/4] 加载数据...")
    gt_data = load_jsonl(args.gt_clues)
    pred_data = load_jsonl(args.pred_clues)
    
    print(f"  GT样本数: {len(gt_data)}")
    print(f"  Pred样本数: {len(pred_data)}")
    
    # 2. 建立name到数据的映射（以GT为准）
    gt_dict = {item['name']: item for item in gt_data}
    pred_dict = {item['name']: item for item in pred_data}
    
    # 找到GT中存在且Pred中也存在的样本
    common_names = [name for name in gt_dict.keys() if name in pred_dict]
    print(f"  匹配样本数: {len(common_names)}")
    
    if len(common_names) == 0:
        print("[ERROR] 没有找到匹配的样本！请检查name字段格式。")
        return
    
    # 3. 加载embedding模型
    print("\n[2/4] 加载Embedding模型...")
    tokenizer, model = load_embedder()
    
    # 4. 逐样本计算指标（核心逻辑）
    print("\n[3/4] 逐样本计算语义召回率和精确率...")
    
    # 统计变量
    total_gt_visual = 0
    total_gt_audio = 0
    total_pred_visual = 0
    total_pred_audio = 0
    
    # 每个样本的指标
    vis_recalls = []
    vis_precisions = []
    aud_recalls = []
    aud_precisions = []
    overall_recalls = []
    overall_precisions = []
    
    # 汇总的匹配数（用于Micro-average）
    vis_matched_gt_sum = 0
    vis_matched_pred_sum = 0
    aud_matched_gt_sum = 0
    aud_matched_pred_sum = 0
    overall_matched_gt_sum = 0
    overall_matched_pred_sum = 0
    
    sample_details = []
    
    for name in tqdm(common_names, desc="逐样本评估"):
        gt_item = gt_dict[name]
        pred_item = pred_dict[name]
        
        # 提取线索
        gt_visual = gt_item.get('visual_cues', [])
        gt_audio = gt_item.get('audio_cues', [])
        pred_visual = pred_item.get('visual_cues', [])
        pred_audio = pred_item.get('audio_cues', [])
        
        # 确保是列表并过滤空字符串
        gt_visual = [c for c in gt_visual if c and isinstance(c, str) and len(c.strip()) > 0] if isinstance(gt_visual, list) else []
        gt_audio = [c for c in gt_audio if c and isinstance(c, str) and len(c.strip()) > 0] if isinstance(gt_audio, list) else []
        pred_visual = [c for c in pred_visual if c and isinstance(c, str) and len(c.strip()) > 0] if isinstance(pred_visual, list) else []
        pred_audio = [c for c in pred_audio if c and isinstance(c, str) and len(c.strip()) > 0] if isinstance(pred_audio, list) else []
        
        # 统计总数
        total_gt_visual += len(gt_visual)
        total_gt_audio += len(gt_audio)
        total_pred_visual += len(pred_visual)
        total_pred_audio += len(pred_audio)
        
        # 编码该样本的线索
        gt_visual_emb = encode_texts(gt_visual, tokenizer, model) if gt_visual else np.array([])
        pred_visual_emb = encode_texts(pred_visual, tokenizer, model) if pred_visual else np.array([])
        gt_audio_emb = encode_texts(gt_audio, tokenizer, model) if gt_audio else np.array([])
        pred_audio_emb = encode_texts(pred_audio, tokenizer, model) if pred_audio else np.array([])
        
        # 计算该样本的视觉指标
        vis_metrics = compute_sample_metrics(gt_visual, pred_visual, gt_visual_emb, pred_visual_emb, args.threshold)
        if vis_metrics['valid_recall']:
            vis_recalls.append(vis_metrics['recall'])
        if vis_metrics['valid_precision']:
            vis_precisions.append(vis_metrics['precision'])
        vis_matched_gt_sum += vis_metrics['matched_gt']
        vis_matched_pred_sum += vis_metrics['matched_pred']
        
        # 计算该样本的音频指标
        aud_metrics = compute_sample_metrics(gt_audio, pred_audio, gt_audio_emb, pred_audio_emb, args.threshold)
        if aud_metrics['valid_recall']:
            aud_recalls.append(aud_metrics['recall'])
        if aud_metrics['valid_precision']:
            aud_precisions.append(aud_metrics['precision'])
        aud_matched_gt_sum += aud_metrics['matched_gt']
        aud_matched_pred_sum += aud_metrics['matched_pred']
        
        # 计算该样本的整体指标（根据方法选择）
        if args.overall_method == "weighted":
            # 方法1: 加权方式（分别计算，严格区分模态）
            # 整体召回 = (视觉匹配数 + 音频匹配数) / (视觉GT总数 + 音频GT总数)
            # 整体精确 = (视觉匹配数 + 音频匹配数) / (视觉Pred总数 + 音频Pred总数)
            total_gt = vis_metrics['total_gt'] + aud_metrics['total_gt']
            total_pred = vis_metrics['total_pred'] + aud_metrics['total_pred']
            matched_gt = vis_metrics['matched_gt'] + aud_metrics['matched_gt']
            matched_pred = vis_metrics['matched_pred'] + aud_metrics['matched_pred']
            
            overall_recall = matched_gt / total_gt if total_gt > 0 else 0.0
            overall_precision = matched_pred / total_pred if total_pred > 0 else 0.0
            
            if total_gt > 0:
                overall_recalls.append(overall_recall)
            if total_pred > 0:
                overall_precisions.append(overall_precision)
            
            overall_matched_gt_sum += matched_gt
            overall_matched_pred_sum += matched_pred
            
        else:  # combined
            # 方法2: 合并embedding方式（允许跨模态匹配）
            gt_combined = gt_visual + gt_audio
            pred_combined = pred_visual + pred_audio
            if len(gt_visual_emb) > 0 and len(gt_audio_emb) > 0:
                gt_combined_emb = np.concatenate([gt_visual_emb, gt_audio_emb], axis=0)
            elif len(gt_visual_emb) > 0:
                gt_combined_emb = gt_visual_emb
            elif len(gt_audio_emb) > 0:
                gt_combined_emb = gt_audio_emb
            else:
                gt_combined_emb = np.array([])
            
            if len(pred_visual_emb) > 0 and len(pred_audio_emb) > 0:
                pred_combined_emb = np.concatenate([pred_visual_emb, pred_audio_emb], axis=0)
            elif len(pred_visual_emb) > 0:
                pred_combined_emb = pred_visual_emb
            elif len(pred_audio_emb) > 0:
                pred_combined_emb = pred_audio_emb
            else:
                pred_combined_emb = np.array([])
            
            overall_metrics = compute_sample_metrics(gt_combined, pred_combined, gt_combined_emb, pred_combined_emb, args.threshold)
            if overall_metrics['valid_recall']:
                overall_recalls.append(overall_metrics['recall'])
            if overall_metrics['valid_precision']:
                overall_precisions.append(overall_metrics['precision'])
            overall_matched_gt_sum += overall_metrics['matched_gt']
            overall_matched_pred_sum += overall_metrics['matched_pred']
            
            overall_recall = overall_metrics['recall']
            overall_precision = overall_metrics['precision']
        
        # 保存样本详情
        sample_details.append({
            'name': name,
            'vis_recall': vis_metrics['recall'],
            'vis_precision': vis_metrics['precision'],
            'aud_recall': aud_metrics['recall'],
            'aud_precision': aud_metrics['precision'],
            'overall_recall': overall_recall,
            'overall_precision': overall_precision,
        })
    
    print(f"\n  GT视觉线索总数: {total_gt_visual}")
    print(f"  GT音频线索总数: {total_gt_audio}")
    print(f"  Pred视觉线索总数: {total_pred_visual}")
    print(f"  Pred音频线索总数: {total_pred_audio}")
    
    # 计算Macro-average（样本级平均）
    print("\n[4/4] 汇总指标...")
    
    vis_recall = np.mean(vis_recalls) if vis_recalls else 0.0
    vis_precision = np.mean(vis_precisions) if vis_precisions else 0.0
    aud_recall = np.mean(aud_recalls) if aud_recalls else 0.0
    aud_precision = np.mean(aud_precisions) if aud_precisions else 0.0
    overall_recall = np.mean(overall_recalls) if overall_recalls else 0.0
    overall_precision = np.mean(overall_precisions) if overall_precisions else 0.0
    
    # 用于显示的统计数据
    vis_total_gt = total_gt_visual
    vis_total_pred = total_pred_visual
    vis_matched_gt = vis_matched_gt_sum
    vis_matched_pred = vis_matched_pred_sum
    
    aud_total_gt = total_gt_audio
    aud_total_pred = total_pred_audio
    aud_matched_gt = aud_matched_gt_sum
    aud_matched_pred = aud_matched_pred_sum
    
    overall_total_gt = total_gt_visual + total_gt_audio
    overall_total_pred = total_pred_visual + total_pred_audio
    overall_matched_gt = overall_matched_gt_sum
    overall_matched_pred = overall_matched_pred_sum
    
    # 计算Micro-average（线索级平均）
    vis_recall_micro = vis_matched_gt_sum / vis_total_gt if vis_total_gt > 0 else 0.0
    vis_precision_micro = vis_matched_pred_sum / vis_total_pred if vis_total_pred > 0 else 0.0
    aud_recall_micro = aud_matched_gt_sum / aud_total_gt if aud_total_gt > 0 else 0.0
    aud_precision_micro = aud_matched_pred_sum / aud_total_pred if aud_total_pred > 0 else 0.0
    overall_recall_micro = overall_matched_gt_sum / overall_total_gt if overall_total_gt > 0 else 0.0
    overall_precision_micro = overall_matched_pred_sum / overall_total_pred if overall_total_pred > 0 else 0.0
    
    # 7. 打印结果
    print("\n" + "=" * 70)
    print("评估结果（逐样本Macro-Average）")
    print("=" * 70)
    
    print("\n【视觉线索 (Visual Claims)】")
    print(f"  GT线索总数:   {vis_total_gt}")
    print(f"  Pred线索总数: {vis_total_pred}")
    print(f"  有效样本数:   召回={len(vis_recalls)}, 精确={len(vis_precisions)}")
    print(f"  Macro-Recall:    {vis_recall:.4f} ({vis_recall*100:.2f}%)")
    print(f"  Macro-Precision: {vis_precision:.4f} ({vis_precision*100:.2f}%)")
    print(f"  Micro-Recall:    {vis_recall_micro:.4f} ({vis_recall_micro*100:.2f}%)")
    print(f"  Micro-Precision: {vis_precision_micro:.4f} ({vis_precision_micro*100:.2f}%)")
    
    print("\n【音频线索 (Audio Claims)】")
    print(f"  GT线索总数:   {aud_total_gt}")
    print(f"  Pred线索总数: {aud_total_pred}")
    print(f"  有效样本数:   召回={len(aud_recalls)}, 精确={len(aud_precisions)}")
    print(f"  Macro-Recall:    {aud_recall:.4f} ({aud_recall*100:.2f}%)")
    print(f"  Macro-Precision: {aud_precision:.4f} ({aud_precision*100:.2f}%)")
    print(f"  Micro-Recall:    {aud_recall_micro:.4f} ({aud_recall_micro*100:.2f}%)")
    print(f"  Micro-Precision: {aud_precision_micro:.4f} ({aud_precision_micro*100:.2f}%)")
    
    print("\n【整体线索 (Overall: Visual + Audio)】")
    print(f"  GT线索总数:   {overall_total_gt}")
    print(f"  Pred线索总数: {overall_total_pred}")
    print(f"  有效样本数:   召回={len(overall_recalls)}, 精确={len(overall_precisions)}")
    print(f"  Macro-Recall:    {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"  Macro-Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"  Micro-Recall:    {overall_recall_micro:.4f} ({overall_recall_micro*100:.2f}%)")
    print(f"  Micro-Precision: {overall_precision_micro:.4f} ({overall_precision_micro*100:.2f}%)")
    
    # 8. 保存结果
    results = {
        'config': {
            'gt_file': args.gt_clues,
            'pred_file': args.pred_clues,
            'threshold': args.threshold,
            'num_samples': len(common_names),
            'method': 'per_sample_macro_average',
            'overall_method': args.overall_method,
        },
        'visual': {
            'gt_total': vis_total_gt,
            'pred_total': vis_total_pred,
            'valid_samples_recall': len(vis_recalls),
            'valid_samples_precision': len(vis_precisions),
            'macro_recall': vis_recall,
            'macro_precision': vis_precision,
            'micro_recall': vis_recall_micro,
            'micro_precision': vis_precision_micro,
        },
        'audio': {
            'gt_total': aud_total_gt,
            'pred_total': aud_total_pred,
            'valid_samples_recall': len(aud_recalls),
            'valid_samples_precision': len(aud_precisions),
            'macro_recall': aud_recall,
            'macro_precision': aud_precision,
            'micro_recall': aud_recall_micro,
            'micro_precision': aud_precision_micro,
        },
        'overall': {
            'gt_total': overall_total_gt,
            'pred_total': overall_total_pred,
            'valid_samples_recall': len(overall_recalls),
            'valid_samples_precision': len(overall_precisions),
            'macro_recall': overall_recall,
            'macro_precision': overall_precision,
            'micro_recall': overall_recall_micro,
            'micro_precision': overall_precision_micro,
        }
    }
    
    output_file = output_dir / f"semantic_recall_results_t{args.threshold}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 同时生成CSV格式的摘要
    csv_file = output_dir / f"semantic_recall_summary_t{args.threshold}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("modality,gt_total,pred_total,macro_recall,macro_precision,micro_recall,micro_precision\n")
        f.write(f"visual,{vis_total_gt},{vis_total_pred},{vis_recall:.4f},{vis_precision:.4f},{vis_recall_micro:.4f},{vis_precision_micro:.4f}\n")
        f.write(f"audio,{aud_total_gt},{aud_total_pred},{aud_recall:.4f},{aud_precision:.4f},{aud_recall_micro:.4f},{aud_precision_micro:.4f}\n")
        f.write(f"overall,{overall_total_gt},{overall_total_pred},{overall_recall:.4f},{overall_precision:.4f},{overall_recall_micro:.4f},{overall_precision_micro:.4f}\n")
    
    print(f"CSV摘要已保存到: {csv_file}")
    
    # 如果开启详细模式，保存逐样本的详细信息
    if args.save_details:
        print(f"\n保存逐样本详情...")
        
        details = {
            'threshold': args.threshold,
            'per_sample_results': sample_details,
        }
        
        details_file = output_dir / f"semantic_recall_details_t{args.threshold}.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        
        print(f"逐样本详细信息已保存到: {details_file}")


if __name__ == "__main__":
    main()
