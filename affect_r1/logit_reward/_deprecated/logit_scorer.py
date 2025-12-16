"""
Logit 打分模块

负责:
1. 计算 GT 情感类别的对比得分 (Contrastive GT Score)
2. 支持 Log-Sum-Exp 聚合
3. 处理类别不平衡问题
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple

from .emotion_vocab import EmotionVocabManager


class LogitScorer:
    """
    基于 Logits 的情感得分计算器
    
    核心公式:
    δ(t) = mean_{c ∈ C_gt} S(c, t) - max_{c ∉ C_gt} S(c, t)
    
    其中 S(c, t) = log(Σ_{w ∈ V(c)} P(w | h_t) + ε)
    """
    
    def __init__(
        self,
        vocab_manager: EmotionVocabManager,
        use_max_for_neg: bool = True,
        epsilon: float = 1e-10,
    ):
        """
        初始化 Logit 打分器
        
        Args:
            vocab_manager: 情绪词表管理器
            use_max_for_neg: 是否对非 GT 类别使用 max（推荐 True，处理类别不平衡）
            epsilon: 防止 log(0) 的小常数
        """
        self.vocab_manager = vocab_manager
        self.use_max_for_neg = use_max_for_neg
        self.epsilon = epsilon
        
        # 预构建类别 -> token_ids 的 tensor 形式（用于批量计算）
        self._category_token_tensors: Dict[str, torch.Tensor] = {}
    
    def _get_category_token_tensor(self, category: str, device: torch.device) -> torch.Tensor:
        """获取类别对应的 token_id tensor（带缓存）"""
        cache_key = f"{category}_{device}"
        if cache_key not in self._category_token_tensors:
            token_ids = list(self.vocab_manager.get_token_ids_for_category(category))
            if token_ids:
                self._category_token_tensors[cache_key] = torch.tensor(
                    token_ids, dtype=torch.long, device=device
                )
            else:
                self._category_token_tensors[cache_key] = None
        return self._category_token_tensors.get(cache_key)
    
    def compute_category_score(
        self,
        logits: torch.Tensor,
        category: str,
    ) -> torch.Tensor:
        """
        计算单个情感类别在给定 logits 下的得分
        
        S(c, t) = log(Σ_{w ∈ V(c)} P(w | h_t) + ε)
        
        Args:
            logits: (batch_size, vocab_size) 或 (vocab_size,) 的 logits
            category: 情感类别名称
        
        Returns:
            score: (batch_size,) 或 scalar 的得分
        """
        is_1d = logits.dim() == 1
        if is_1d:
            logits = logits.unsqueeze(0)
        
        device = logits.device
        token_ids = self._get_category_token_tensor(category, device)
        
        if token_ids is None or len(token_ids) == 0:
            # 该类别没有有效的 token，返回极小值
            result = torch.full((logits.size(0),), -100.0, device=device)
            return result.squeeze(0) if is_1d else result
        
        # 计算 softmax 概率
        probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)
        
        # 取出该类别所有 token 的概率并求和
        category_probs = probs[:, token_ids]  # (batch_size, num_tokens)
        sum_probs = category_probs.sum(dim=-1)  # (batch_size,)
        
        # Log-Sum-Exp
        score = torch.log(sum_probs + self.epsilon)
        
        return score.squeeze(0) if is_1d else score
    
    def compute_contrastive_score(
        self,
        logits: torch.Tensor,
        gt_categories: List[str],
        all_categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算对比情感得分
        
        δ(t) = mean_{c ∈ C_gt} S(c, t) - max_{c ∉ C_gt} S(c, t)
        
        Args:
            logits: (batch_size, vocab_size) 或 (vocab_size,) 的 logits
            gt_categories: Ground Truth 情感类别列表
            all_categories: 所有候选类别，默认使用词表中的所有类别
        
        Returns:
            delta: (batch_size,) 或 scalar 的对比得分
        """
        is_1d = logits.dim() == 1
        if is_1d:
            logits = logits.unsqueeze(0)
        
        device = logits.device
        batch_size = logits.size(0)
        
        if all_categories is None:
            all_categories = self.vocab_manager.get_all_categories()
        
        # 归一化 GT 类别名称
        gt_set = set(c.lower().strip() for c in gt_categories)
        neg_categories = [c for c in all_categories if c.lower().strip() not in gt_set]
        
        # 计算 GT 类别得分的均值
        gt_scores = []
        for cat in gt_categories:
            cat = cat.lower().strip()
            score = self.compute_category_score(logits, cat)
            gt_scores.append(score)
        
        if gt_scores:
            gt_scores_tensor = torch.stack(gt_scores, dim=-1)  # (batch_size, num_gt)
            gt_mean = gt_scores_tensor.mean(dim=-1)  # (batch_size,)
        else:
            gt_mean = torch.zeros(batch_size, device=device)
        
        # 计算非 GT 类别得分
        neg_scores = []
        for cat in neg_categories:
            cat = cat.lower().strip()
            score = self.compute_category_score(logits, cat)
            neg_scores.append(score)
        
        if neg_scores:
            neg_scores_tensor = torch.stack(neg_scores, dim=-1)  # (batch_size, num_neg)
            if self.use_max_for_neg:
                # 使用 max 处理类别不平衡
                neg_agg = neg_scores_tensor.max(dim=-1).values  # (batch_size,)
            else:
                neg_agg = neg_scores_tensor.mean(dim=-1)  # (batch_size,)
        else:
            neg_agg = torch.zeros(batch_size, device=device)
        
        # 对比得分
        delta = gt_mean - neg_agg
        
        return delta.squeeze(0) if is_1d else delta
    
    def compute_sequence_score(
        self,
        logits_seq: torch.Tensor,
        emotion_mask: torch.Tensor,
        gt_categories: List[str],
        all_categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算整个序列的聚合得分
        
        S_seq = Σ_t (m_t · δ_t) / (Σ_t m_t + ε)
        
        Args:
            logits_seq: (batch_size, seq_len, vocab_size) 的 logits 序列
            emotion_mask: (batch_size, seq_len) 的情感词 mask
            gt_categories: Ground Truth 情感类别列表
            all_categories: 所有候选类别
        
        Returns:
            seq_score: (batch_size,) 的序列得分
        """
        batch_size, seq_len, vocab_size = logits_seq.shape
        device = logits_seq.device
        
        # 逐 token 计算对比得分
        deltas = torch.zeros(batch_size, seq_len, device=device)
        
        for t in range(seq_len):
            logits_t = logits_seq[:, t, :]  # (batch_size, vocab_size)
            delta_t = self.compute_contrastive_score(
                logits_t, gt_categories, all_categories
            )  # (batch_size,)
            deltas[:, t] = delta_t
        
        # 加权聚合
        masked_deltas = deltas * emotion_mask.float()  # (batch_size, seq_len)
        mask_sum = emotion_mask.float().sum(dim=-1) + self.epsilon  # (batch_size,)
        seq_score = masked_deltas.sum(dim=-1) / mask_sum  # (batch_size,)
        
        return seq_score
    
    def compute_batch_sequence_scores(
        self,
        logits_seq: torch.Tensor,
        answer_token_ids: torch.Tensor,
        gt_categories_list: List[List[str]],
        skip_tokens: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        """
        批量计算多个样本的序列得分
        
        Args:
            logits_seq: (batch_size, seq_len, vocab_size) 的 logits 序列
            answer_token_ids: (batch_size, seq_len) 的 answer token ids
            gt_categories_list: 每个样本的 GT 类别列表
            skip_tokens: 跳过的 token ids
        
        Returns:
            scores: (batch_size,) 的得分
        """
        batch_size = logits_seq.size(0)
        device = logits_seq.device
        scores = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            # 构建该样本的 emotion mask
            mask = self.vocab_manager.build_answer_emotion_mask(
                answer_token_ids[b], skip_tokens
            )  # (seq_len,)
            
            # 计算序列得分
            score = self.compute_sequence_score(
                logits_seq[b:b+1],  # (1, seq_len, vocab_size)
                mask.unsqueeze(0),  # (1, seq_len)
                gt_categories_list[b],
            )  # (1,)
            scores[b] = score.squeeze(0)
        
        return scores
