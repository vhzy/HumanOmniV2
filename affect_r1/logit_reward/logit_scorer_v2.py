"""
Logit 打分模块 V2

Phase 1: 在线训练流程

功能：
1. 动态锚点识别 - 构建 Reward Calculation Mask
2. 双路 Forward 计算 - With-Think / No-Think
3. 信号计算 - Log-Sum-Exp 聚合 + 增益差分
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple

try:
    from .emotion_token_map import EmotionTokenMap
except ImportError:
    from emotion_token_map import EmotionTokenMap


class LogitScorerV2:
    """
    基于 Logits 的情感得分计算器 V2
    
    实现 Phase 1 中的信号计算逻辑：
    
    1. 聚合置信度:
       S(c, t) = log(Σ_{id ∈ SafeTokens(c)} P(token_id | h_t) + ε)
    
    2. 增益差分:
       Gain(t) = [S(GT, t)_with - S(GT, t)_no] - [S(Neg, t)_with - S(Neg, t)_no]
    
    3. 序列平均:
       R_final = (1 / Σ M_t) * Σ_t (M_t · Gain(t))
    """
    
    def __init__(
        self,
        token_map: EmotionTokenMap,
        epsilon: float = 1e-10,
        use_max_for_neg: bool = False,
    ):
        """
        Args:
            token_map: 情感 Token 映射表
            epsilon: 防止 log(0) 的小常数
            use_max_for_neg: 非 GT 类别是否使用 max 聚合
        """
        self.token_map = token_map
        self.epsilon = epsilon
        self.use_max_for_neg = use_max_for_neg
        
        # 缓存 category -> token tensor
        self._category_token_cache: Dict[str, torch.Tensor] = {}
    
    def _get_category_tokens_tensor(
        self, 
        category: str, 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """获取类别对应的安全 Token ID tensor（带缓存）"""
        cache_key = f"{category}_{device}"
        
        if cache_key not in self._category_token_cache:
            token_ids = self.token_map.get_safe_tokens_for_category(category)
            if token_ids:
                self._category_token_cache[cache_key] = torch.tensor(
                    list(token_ids), dtype=torch.long, device=device
                )
            else:
                self._category_token_cache[cache_key] = None
        
        return self._category_token_cache.get(cache_key)
    
    def compute_category_score(
        self,
        logits: torch.Tensor,
        category: str,
    ) -> torch.Tensor:
        """
        计算单个情感类别的聚合置信度
        
        S(c, t) = log(Σ_{id ∈ SafeTokens(c)} P(token_id | h_t) + ε)
        
        Args:
            logits: (batch_size, vocab_size) 或 (vocab_size,)
            category: 情感类别名
        
        Returns:
            score: (batch_size,) 或 scalar
        """
        is_1d = logits.dim() == 1
        if is_1d:
            logits = logits.unsqueeze(0)
        
        device = logits.device
        token_ids = self._get_category_tokens_tensor(category, device)
        
        if token_ids is None or len(token_ids) == 0:
            result = torch.full((logits.size(0),), -100.0, device=device)
            return result.squeeze(0) if is_1d else result
        
        # softmax 概率
        probs = F.softmax(logits, dim=-1)
        
        # 取出该类别所有 safe token 的概率并求和
        category_probs = probs[:, token_ids]
        sum_probs = category_probs.sum(dim=-1)
        
        # Log-Sum-Exp
        score = torch.log(sum_probs + self.epsilon)
        
        return score.squeeze(0) if is_1d else score
    
    def compute_gt_score(
        self,
        logits: torch.Tensor,
        gt_categories: List[str],
    ) -> torch.Tensor:
        """
        计算 GT 类别的聚合得分
        
        S(GT, t) = mean_{c ∈ C_gt} S(c, t)
        
        Args:
            logits: (batch_size, vocab_size)
            gt_categories: GT 情感类别列表
        
        Returns:
            score: (batch_size,)
        """
        if not gt_categories:
            return torch.zeros(logits.size(0), device=logits.device)
        
        scores = []
        for cat in gt_categories:
            score = self.compute_category_score(logits, cat)
            scores.append(score)
        
        scores_tensor = torch.stack(scores, dim=-1)
        return scores_tensor.mean(dim=-1)
    
    def compute_neg_score(
        self,
        logits: torch.Tensor,
        gt_categories: List[str],
        all_categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算非 GT 类别的聚合得分
        
        S(Neg, t) = max_{c ∉ C_gt} S(c, t)  或  mean_{c ∉ C_gt} S(c, t)
        
        Args:
            logits: (batch_size, vocab_size)
            gt_categories: GT 情感类别列表
            all_categories: 所有候选类别
        
        Returns:
            score: (batch_size,)
        """
        if all_categories is None:
            all_categories = self.token_map.get_all_categories()
        
        gt_set = set(c.lower().strip() for c in gt_categories)
        neg_categories = [c for c in all_categories if c.lower().strip() not in gt_set]
        
        if not neg_categories:
            return torch.zeros(logits.size(0), device=logits.device)
        
        scores = []
        for cat in neg_categories:
            score = self.compute_category_score(logits, cat)
            scores.append(score)
        
        scores_tensor = torch.stack(scores, dim=-1)
        
        if self.use_max_for_neg:
            return scores_tensor.max(dim=-1).values
        else:
            return scores_tensor.mean(dim=-1)
    
    # ================================================================
    # 按 Wheel 分开计算的方法（核心）
    # ================================================================
    
    def compute_wheel_l1_score(
        self,
        logits: torch.Tensor,
        wheel_name: str,
        l1_category: str,
    ) -> torch.Tensor:
        """计算指定情绪轮的指定 Level1 类别的得分"""
        is_1d = logits.dim() == 1
        if is_1d:
            logits = logits.unsqueeze(0)
        
        device = logits.device
        token_ids = self.token_map.get_wheel_l1_safe_tokens(wheel_name, l1_category)
        
        if not token_ids:
            result = torch.full((logits.size(0),), -100.0, device=device)
            return result.squeeze(0) if is_1d else result
        
        token_tensor = torch.tensor(list(token_ids), dtype=torch.long, device=device)
        probs = F.softmax(logits, dim=-1)
        category_probs = probs[:, token_tensor]
        sum_probs = category_probs.sum(dim=-1)
        score = torch.log(sum_probs + self.epsilon)
        
        return score.squeeze(0) if is_1d else score
    
    def compute_wheel_gain(
        self,
        logits_with: torch.Tensor,
        logits_no: torch.Tensor,
        wheel_name: str,
        gt_l1_categories: Set[str],
    ) -> torch.Tensor:
        """计算单个情绪轮内的增益差分"""
        device = logits_with.device
        all_l1 = self.token_map.get_wheel_level1_categories(wheel_name)
        
        if not all_l1 or not gt_l1_categories:
            return torch.zeros(logits_with.size(0), device=device)
        
        # GT 类别得分
        gt_scores_with = []
        gt_scores_no = []
        for l1 in gt_l1_categories:
            gt_scores_with.append(self.compute_wheel_l1_score(logits_with, wheel_name, l1))
            gt_scores_no.append(self.compute_wheel_l1_score(logits_no, wheel_name, l1))
        
        s_gt_with = torch.stack(gt_scores_with, dim=-1).mean(dim=-1)
        s_gt_no = torch.stack(gt_scores_no, dim=-1).mean(dim=-1)
        
        # Neg 类别得分
        neg_l1 = [l1 for l1 in all_l1 if l1 not in gt_l1_categories]
        
        if not neg_l1:
            return s_gt_with - s_gt_no
        
        neg_scores_with = []
        neg_scores_no = []
        for l1 in neg_l1:
            neg_scores_with.append(self.compute_wheel_l1_score(logits_with, wheel_name, l1))
            neg_scores_no.append(self.compute_wheel_l1_score(logits_no, wheel_name, l1))
        
        neg_tensor_with = torch.stack(neg_scores_with, dim=-1)
        neg_tensor_no = torch.stack(neg_scores_no, dim=-1)
        
        if self.use_max_for_neg:
            s_neg_with = neg_tensor_with.max(dim=-1).values
            s_neg_no = neg_tensor_no.max(dim=-1).values
        else:
            s_neg_with = neg_tensor_with.mean(dim=-1)
            s_neg_no = neg_tensor_no.mean(dim=-1)
        
        gain = (s_gt_with - s_gt_no) - (s_neg_with - s_neg_no)
        return gain
    
    def compute_multi_wheel_gain(
        self,
        logits_with: torch.Tensor,
        logits_no: torch.Tensor,
        gt_words: List[str],
    ) -> torch.Tensor:
        """计算多个情绪轮的平均增益"""
        device = logits_with.device
        
        wheel_to_l1 = self.token_map.find_gt_in_wheels(gt_words)
        
        if not wheel_to_l1:
            return torch.zeros(logits_with.size(0), device=device)
        
        wheel_gains = []
        for wheel_name, gt_l1_set in wheel_to_l1.items():
            gain = self.compute_wheel_gain(logits_with, logits_no, wheel_name, gt_l1_set)
            wheel_gains.append(gain)
        
        gains_tensor = torch.stack(wheel_gains, dim=-1)
        return gains_tensor.mean(dim=-1)
    
    def compute_position_gain(
        self,
        logits_with: torch.Tensor,
        logits_no: torch.Tensor,
        gt_categories: List[str],
        all_categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算单个位置的增益差分（使用多 wheel 计算）
        
        GT 可能映射到多个情绪轮，分别在每个轮内计算后平均
        
        Args:
            logits_with: With-Think 的 logits (batch_size, vocab_size)
            logits_no: No-Think 的 logits (batch_size, vocab_size)
            gt_categories: GT 情感词列表
            all_categories: 忽略（保留兼容）
        
        Returns:
            gain: (batch_size,)
        """
        # 使用新的多 wheel 计算方式
        return self.compute_multi_wheel_gain(logits_with, logits_no, gt_categories)
    
    def compute_sequence_reward(
        self,
        logits_seq_with: torch.Tensor,
        logits_seq_no: torch.Tensor,
        anchor_mask: torch.Tensor,
        gt_categories: List[str],
        all_categories: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算序列的最终 Reward
        
        R_final = (1 / Σ M_t) * Σ_t (M_t · Gain(t))
        
        Args:
            logits_seq_with: With-Think 的 logits 序列 (seq_len, vocab_size)
            logits_seq_no: No-Think 的 logits 序列 (seq_len, vocab_size)
            anchor_mask: 锚点 mask (seq_len,)
            gt_categories: GT 情感类别
            all_categories: 所有候选类别
        
        Returns:
            reward: scalar
        """
        seq_len = logits_seq_with.size(0)
        device = logits_seq_with.device
        
        gains = torch.zeros(seq_len, device=device)
        
        for t in range(seq_len):
            if anchor_mask[t] == 0:
                continue
            
            gain = self.compute_position_gain(
                logits_seq_with[t:t+1],
                logits_seq_no[t:t+1],
                gt_categories,
                all_categories,
            )
            gains[t] = gain.squeeze()
        
        # 加权平均
        masked_gains = gains * anchor_mask.float()
        mask_sum = anchor_mask.float().sum() + self.epsilon
        reward = masked_gains.sum() / mask_sum
        
        return reward
    
    def compute_batch_rewards(
        self,
        logits_seq_with_list: List[torch.Tensor],
        logits_seq_no_list: List[torch.Tensor],
        answer_token_ids_list: List[torch.Tensor],
        gt_categories_list: List[List[str]],
    ) -> torch.Tensor:
        """
        批量计算多个样本的 Reward
        
        Args:
            logits_seq_with_list: 每个样本的 With-Think logits
            logits_seq_no_list: 每个样本的 No-Think logits
            answer_token_ids_list: 每个样本的 answer token ids
            gt_categories_list: 每个样本的 GT 类别
        
        Returns:
            rewards: (batch_size,)
        """
        batch_size = len(logits_seq_with_list)
        device = logits_seq_with_list[0].device
        rewards = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            # 构建锚点 mask
            anchor_mask = self.token_map.build_anchor_mask(answer_token_ids_list[b])
            
            # 计算 reward
            reward = self.compute_sequence_reward(
                logits_seq_with_list[b],
                logits_seq_no_list[b],
                anchor_mask,
                gt_categories_list[b],
            )
            rewards[b] = reward
        
        return rewards


def create_logit_scorer(
    token_map: EmotionTokenMap,
    epsilon: float = 1e-10,
    use_max_for_neg: bool = False,
) -> LogitScorerV2:
    """
    创建 Logit 打分器
    
    Args:
        token_map: 情感 Token 映射表
        epsilon: 防止 log(0) 的小常数
        use_max_for_neg: 非 GT 类别是否使用 max 聚合
    
    Returns:
        LogitScorerV2 实例
    """
    return LogitScorerV2(
        token_map=token_map,
        epsilon=epsilon,
        use_max_for_neg=use_max_for_neg,
    )
