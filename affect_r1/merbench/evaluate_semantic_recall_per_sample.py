#!/usr/bin/env python3
"""
语义感知质量评估脚本 - 保存每个样本的多模态Recall

功能：
1. 加载GT和Pred的线索提取结果
2. 使用Qwen3-embedding-0.6B计算语义相似度
3. 计算每个样本的视觉/音频/整体线索的召回率
4. 保存每个样本的结果到JSONL文件

使用示例：
CUDA_VISIBLE_DEVICES=1 python -m affect_r1.merbench.evaluate_semantic_recall_per_sample \
    --gt-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl \
    --pred-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_baseline6/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/clue_extraction_Qwen.jsonl \
    --threshold 0.6
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
            
            # L2归一化（处理零向量情况）
            norms = embeddings.norm(dim=1, keepdim=True)
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            embeddings = embeddings / norms
            
            # 将 NaN 替换为 0
            if torch.isnan(embeddings).any():
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    result = np.concatenate(all_embeddings, axis=0)
    
    # 最终检查
    if np.isnan(result).sum() > 0:
        result = np.nan_to_num(result, nan=0.0)
    
    return result


def compute_sample_metrics(
    gt_cues: List[str],
    pred_cues: List[str],
    gt_emb: np.ndarray,
    pred_emb: np.ndarray,
    threshold: float = 0.7
) -> Dict:
    """
    计算单个样本的召回率和精确率
    """
    total_gt = len(gt_cues)
    total_pred = len(pred_cues)
    
    # 处理边界情况
    if total_gt == 0 and total_pred == 0:
        return {
            'recall': 1.0, 'precision': 1.0,
            'matched_gt': 0, 'total_gt': 0,
            'matched_pred': 0, 'total_pred': 0,
            'valid_recall': False, 'valid_precision': False
        }
    
    if total_gt == 0:
        return {
            'recall': 1.0, 'precision': 0.0,
            'matched_gt': 0, 'total_gt': 0,
            'matched_pred': 0, 'total_pred': total_pred,
            'valid_recall': False, 'valid_precision': True
        }
    
    if total_pred == 0:
        return {
            'recall': 0.0, 'precision': 1.0,
            'matched_gt': 0, 'total_gt': total_gt,
            'matched_pred': 0, 'total_pred': 0,
            'valid_recall': True, 'valid_precision': False
        }
    
    # 正常情况：都不为空
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
    max_sim_gt = sim_matrix.max(axis=1)
    matched_gt = int((max_sim_gt >= threshold).sum())
    recall = matched_gt / total_gt
    
    # 精确率：对于每个Pred，找GT中最相似的
    max_sim_pred = sim_matrix.max(axis=0)
    matched_pred = int((max_sim_pred >= threshold).sum())
    precision = matched_pred / total_pred
    
    return {
        'recall': recall, 'precision': precision,
        'matched_gt': matched_gt, 'total_gt': total_gt,
        'matched_pred': matched_pred, 'total_pred': total_pred,
        'valid_recall': True, 'valid_precision': True
    }


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
    parser = argparse.ArgumentParser(description="语义感知质量评估 - 保存每个样本的多模态Recall")
    parser.add_argument("--gt-clues", type=str, required=True,
                        help="GT线索文件路径 (JSONL格式)")
    parser.add_argument("--pred-clues", type=str, required=True,
                        help="Pred线索文件路径")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="相似度阈值 (默认0.7)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认为pred文件所在目录)")
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.pred_clues).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("语义感知质量评估 - 保存每个样本的多模态Recall")
    print("=" * 60)
    print(f"GT文件: {args.gt_clues}")
    print(f"Pred文件: {args.pred_clues}")
    print(f"相似度阈值: {args.threshold}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 加载数据
    print("[1/4] 加载数据...")
    gt_data = load_jsonl(args.gt_clues)
    pred_data = load_jsonl(args.pred_clues)
    
    print(f"  GT样本数: {len(gt_data)}")
    print(f"  Pred样本数: {len(pred_data)}")
    
    # 2. 建立name到数据的映射
    gt_dict = {item['name']: item for item in gt_data}
    pred_dict = {item['name']: item for item in pred_data}
    
    common_names = [name for name in gt_dict.keys() if name in pred_dict]
    print(f"  匹配样本数: {len(common_names)}")
    
    if len(common_names) == 0:
        print("[ERROR] 没有找到匹配的样本！")
        return
    
    # 3. 加载embedding模型
    print("\n[2/4] 加载Embedding模型...")
    tokenizer, model = load_embedder()
    
    # 4. 逐样本计算指标
    print("\n[3/4] 逐样本计算语义召回率...")
    
    # 每个样本的详细结果
    per_sample_results = []
    
    # 汇总统计
    vis_recalls = []
    aud_recalls = []
    overall_recalls = []
    
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
        
        # 编码该样本的线索
        gt_visual_emb = encode_texts(gt_visual, tokenizer, model) if gt_visual else np.array([])
        pred_visual_emb = encode_texts(pred_visual, tokenizer, model) if pred_visual else np.array([])
        gt_audio_emb = encode_texts(gt_audio, tokenizer, model) if gt_audio else np.array([])
        pred_audio_emb = encode_texts(pred_audio, tokenizer, model) if pred_audio else np.array([])
        
        # 计算该样本的视觉指标
        vis_metrics = compute_sample_metrics(gt_visual, pred_visual, gt_visual_emb, pred_visual_emb, args.threshold)
        
        # 计算该样本的音频指标
        aud_metrics = compute_sample_metrics(gt_audio, pred_audio, gt_audio_emb, pred_audio_emb, args.threshold)
        
        # 计算该样本的整体/多模态指标 (weighted方式)
        total_gt = vis_metrics['total_gt'] + aud_metrics['total_gt']
        total_pred = vis_metrics['total_pred'] + aud_metrics['total_pred']
        matched_gt = vis_metrics['matched_gt'] + aud_metrics['matched_gt']
        matched_pred = vis_metrics['matched_pred'] + aud_metrics['matched_pred']
        
        multimodal_recall = matched_gt / total_gt if total_gt > 0 else 0.0
        multimodal_precision = matched_pred / total_pred if total_pred > 0 else 0.0
        
        # 收集recall用于平均
        if vis_metrics['valid_recall']:
            vis_recalls.append(vis_metrics['recall'])
        if aud_metrics['valid_recall']:
            aud_recalls.append(aud_metrics['recall'])
        if total_gt > 0:
            overall_recalls.append(multimodal_recall)
        
        # 保存每个样本的结果
        sample_result = {
            'name': name,
            # 视觉模态
            'visual_recall': vis_metrics['recall'],
            'visual_precision': vis_metrics['precision'],
            'visual_matched_gt': vis_metrics['matched_gt'],
            'visual_total_gt': vis_metrics['total_gt'],
            'visual_matched_pred': vis_metrics['matched_pred'],
            'visual_total_pred': vis_metrics['total_pred'],
            # 音频模态
            'audio_recall': aud_metrics['recall'],
            'audio_precision': aud_metrics['precision'],
            'audio_matched_gt': aud_metrics['matched_gt'],
            'audio_total_gt': aud_metrics['total_gt'],
            'audio_matched_pred': aud_metrics['matched_pred'],
            'audio_total_pred': aud_metrics['total_pred'],
            # 多模态整体
            'multimodal_recall': multimodal_recall,
            'multimodal_precision': multimodal_precision,
            'multimodal_matched_gt': matched_gt,
            'multimodal_total_gt': total_gt,
            'multimodal_matched_pred': matched_pred,
            'multimodal_total_pred': total_pred,
        }
        
        per_sample_results.append(sample_result)
    
    # 5. 保存结果
    print("\n[4/4] 保存结果...")
    
    # 保存每个样本的结果到JSONL文件
    per_sample_file = output_dir / f"semantic_recall_per_sample_t{args.threshold}.jsonl"
    with open(per_sample_file, 'w', encoding='utf-8') as f:
        for result in per_sample_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"每个样本的多模态Recall已保存到: {per_sample_file}")
    
    # 计算并打印汇总统计
    print("\n" + "=" * 60)
    print("汇总统计 (Macro-Average)")
    print("=" * 60)
    
    avg_vis_recall = np.mean(vis_recalls) if vis_recalls else 0.0
    avg_aud_recall = np.mean(aud_recalls) if aud_recalls else 0.0
    avg_multimodal_recall = np.mean(overall_recalls) if overall_recalls else 0.0
    
    print(f"\n视觉Recall (有效样本数: {len(vis_recalls)}): {avg_vis_recall:.4f} ({avg_vis_recall*100:.2f}%)")
    print(f"音频Recall (有效样本数: {len(aud_recalls)}): {avg_aud_recall:.4f} ({avg_aud_recall*100:.2f}%)")
    print(f"多模态Recall (有效样本数: {len(overall_recalls)}): {avg_multimodal_recall:.4f} ({avg_multimodal_recall*100:.2f}%)")
    
    # 保存汇总统计到CSV
    summary_file = output_dir / f"semantic_recall_summary_per_sample_t{args.threshold}.csv"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("modality,valid_samples,macro_recall\n")
        f.write(f"visual,{len(vis_recalls)},{avg_vis_recall:.4f}\n")
        f.write(f"audio,{len(aud_recalls)},{avg_aud_recall:.4f}\n")
        f.write(f"multimodal,{len(overall_recalls)},{avg_multimodal_recall:.4f}\n")
    
    print(f"\n汇总统计已保存到: {summary_file}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()

