"""
Logit Reward 计算器 V2

负责:
1. 三路 Forward 的编排 (Full / Perception Counterfactual / Coherence Counterfactual)
2. 计算 Coherence Reward 和 Perception Reward
3. 与 GRPO Trainer 的接口对接

更新:
- 使用新的 EmotionTokenMap (按 wheel 分组的 Token 映射)
- 使用新的 LogitScorerV2 (多 wheel 计算)
"""

import os
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# 使用新的模块
try:
    from .emotion_token_map import EmotionTokenMap, create_emotion_token_map
    from .logit_scorer_v2 import LogitScorerV2, create_logit_scorer
except ImportError:
    from emotion_token_map import EmotionTokenMap, create_emotion_token_map
    from logit_scorer_v2 import LogitScorerV2, create_logit_scorer


@dataclass
class LogitRewardConfig:
    """Logit Reward 配置"""
    use_coherence_reward: bool = True   # 是否使用 coherence reward
    use_perception_reward: bool = True  # 是否使用 perception reward
    alpha: float = 0.1                  # coherence reward 权重
    beta: float = 0.1                   # perception reward 权重
    use_max_for_neg: bool = False        # 非 GT 类别是否使用 max 聚合
    normalize_rewards: bool = False     # 是否归一化 reward
    mapping_depth: str = "full"         # 映射深度: wheel_only, wheel_synonym, full
    cache_path: Optional[str] = None    # Token 映射表缓存路径


class LogitRewardComputer:
    """
    Logit-based Reward 计算器 V2
    
    执行三路 Forward 计算:
    - Forward A: Full State (完整上下文)
    - Forward B: Perception Counterfactual (遮蔽视觉/音频)
    - Forward C: Coherence Counterfactual (移除推理过程)
    
    计算方式:
    - 使用多 wheel 计算：GT 映射到 n 个 wheel，分别计算后平均
    - Gain(t) = [S(GT)_with - S(GT)_no] - [S(Neg)_with - S(Neg)_no]
    
    最终 Reward:
    - r_coh = Gain_Full - Gain_NoThink (推理有效性)
    - r_prcp = Gain_Full - Gain_NoMM   (感知依赖性)
    """
    
    def __init__(
        self,
        token_map: EmotionTokenMap,
        config: Optional[LogitRewardConfig] = None,
        tokenizer=None,
    ):
        """
        初始化 Reward 计算器
        
        Args:
            token_map: 情感 Token 映射表
            config: 配置对象
            tokenizer: tokenizer
        """
        self.token_map = token_map
        self.config = config or LogitRewardConfig()
        self.tokenizer = tokenizer or token_map.tokenizer
        
        # 使用新的 LogitScorerV2
        self.scorer = create_logit_scorer(
            token_map=token_map,
            use_max_for_neg=self.config.use_max_for_neg,
        )
        
        # 缓存 special token ids
        self._think_start_ids = None
        self._think_end_ids = None
        self._answer_start_ids = None
        self._answer_end_ids = None
        self._init_special_token_ids()
    
    def _init_special_token_ids(self):
        """
        初始化 special token ids
        
        注意：tokenizer 可能会将 <tag> 与前面的字符合并，
        所以我们匹配关键部分：'answer>' 和 '</answer'
        """
        # 完整标签编码（用于 _build_no_think_input）
        self._think_start_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        self._think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        self._answer_start_ids = self.tokenizer.encode("<answer>", add_special_tokens=False)
        self._answer_end_ids = self.tokenizer.encode("</answer>", add_special_tokens=False)
        
        # 备用：单独的关键词（用于灵活匹配）
        self._answer_word = self.tokenizer.encode("answer", add_special_tokens=False)
        self._think_word = self.tokenizer.encode("think", add_special_tokens=False)
        self._close_bracket = self.tokenizer.encode(">", add_special_tokens=False)
    
    def _find_pattern_position(
        self,
        token_ids: torch.Tensor,
        pattern: List[int],
    ) -> int:
        """在 token 序列中查找 pattern 的起始位置"""
        seq_len = token_ids.size(0)
        pattern_len = len(pattern)
        
        if seq_len < pattern_len:
            return -1
        
        pattern_tensor = torch.tensor(pattern, dtype=token_ids.dtype, device=token_ids.device)
        
        for i in range(seq_len - pattern_len + 1):
            if torch.equal(token_ids[i:i+pattern_len], pattern_tensor):
                return i
        
        return -1
    
    def _find_all_answer_tags(self, token_ids: torch.Tensor) -> Tuple[List[int], List[int]]:
        """
        找到所有 <answer> 和 </answer> 的位置
        
        策略：
        1. 找所有 'answer' token
        2. 检查前后 token 判断是 <answer> 还是 </answer>
        
        Returns:
            (start_positions, end_positions): 
            - start_positions: 每个 <answer> 之后内容开始的位置
            - end_positions: 每个 </answer> 开始的位置
        """
        seq_len = token_ids.size(0)
        start_positions = []
        end_positions = []
        
        for i in range(seq_len):
            decoded = self.tokenizer.decode([token_ids[i].item()])
            decoded_lower = decoded.lower().strip()
            
            # 检查是否是 'answer' token
            if decoded_lower == 'answer':
                # 检查前一个 token
                prev_decoded = ""
                if i > 0:
                    prev_decoded = self.tokenizer.decode([token_ids[i-1].item()])
                
                # 检查后一个 token
                next_decoded = ""
                if i + 1 < seq_len:
                    next_decoded = self.tokenizer.decode([token_ids[i+1].item()])
                
                # 判断是 <answer> 还是 </answer>
                # <answer>: 前面是 '<' 或以 '<' 结尾，后面以 '>' 开头
                # </answer>: 前面是 '</' 或以 '</' 结尾
                
                is_open_tag = False
                is_close_tag = False
                
                # 检查是否是 </answer>
                if '</' in prev_decoded or prev_decoded.endswith('/'):
                    is_close_tag = True
                # 检查是否是 <answer>
                elif prev_decoded.endswith('<') or prev_decoded.strip() == '<':
                    # 确保不是 </，需要检查前面不是 /
                    if not prev_decoded.endswith('</') and not prev_decoded.endswith('/'):
                        is_open_tag = True
                elif '><' in prev_decoded:
                    # 如 '><' token，说明是 ><answer>
                    is_open_tag = True
                
                # 根据标签类型添加位置
                if is_open_tag and next_decoded.startswith('>'):
                    # <answer> 标签
                    # 检查 '>' 后面是否有其他内容（如 '>worry' tokenized 成 '>w'）
                    if len(next_decoded.strip()) > 1:
                        # '>' 和 answer 内容的第一个字符合并了，需要包含这个 token
                        start_positions.append(i + 1)  # 从 '>w...' 开始
                    else:
                        # '>' 是单独的 token，内容从下一个 token 开始
                        start_positions.append(i + 2)  # 跳过 'answer' 和 '>'
                elif is_close_tag:
                    # </answer> 标签
                    # 找到 '</' 的位置
                    end_pos = i - 1  # '</' token 的位置
                    # 如果前一个 token 包含 '</' 但还有其他内容，需要特殊处理
                    if '</' in prev_decoded and prev_decoded != '</':
                        # 如 '.</' 的情况，</answer> 的内容结束在这个 token 之前的文本部分
                        end_pos = i - 1
                    end_positions.append(end_pos)
        
        return start_positions, end_positions
    
    def _extract_answer_region(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        提取最后一个 <answer>...</answer> 区域的位置
        
        注意：prompt 中可能包含 <answer></answer> 示例，
        所以我们需要找最后一个匹配的 <answer>...</answer> 对
        
        Returns:
            (start, end): answer 内容的起始和结束位置
        """
        start_positions, end_positions = self._find_all_answer_tags(token_ids)
        
        if not start_positions or not end_positions:
            return -1, -1
        
        # 取最后一个 <answer> 和最后一个 </answer>
        # 它们应该是匹配的一对
        last_start = start_positions[-1]
        last_end = end_positions[-1]
        
        # 验证：end 应该在 start 之后
        if last_end <= last_start:
            # 可能有配对问题，尝试从后往前找匹配
            for end in reversed(end_positions):
                for start in reversed(start_positions):
                    if end > start:
                        return start, end
            return -1, -1
        
        return last_start, last_end
    
    def _build_no_think_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建移除推理过程的输入（用于 Coherence Counterfactual）
        
        策略：用正则表达式替换 <think>...</think> 为 <think></think>
        
        原始：... <think>长推理内容</think><answer>...
        结果：... <think></think><answer>...
        """
        import re
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        new_input_ids_list = []
        new_attention_mask_list = []
        
        for b in range(batch_size):
            ids = input_ids[b]
            mask = attention_mask[b]
            
            # 1. Decode 成文本
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            
            # 2. 用正则表达式替换最后一个 <think>...</think> 为 <think></think>
            # 注意：prompt 中可能有 <think> </think> 示例，我们需要替换最后一个（assistant 回复中的）
            pattern = r'<think>.*?</think>'
            matches = list(re.finditer(pattern, text, flags=re.DOTALL))
            
            if matches:
                # 替换最后一个匹配
                last_match = matches[-1]
                new_text = text[:last_match.start()] + '<think></think>' + text[last_match.end():]
            else:
                new_text = text
            
            # 3. 检查是否有替换发生
            if new_text == text:
                # 没有找到 think 块，保持原样
                new_input_ids_list.append(ids)
                new_attention_mask_list.append(mask)
                continue
            
            # 4. Encode 回 token ids
            new_ids = self.tokenizer.encode(new_text, add_special_tokens=False, return_tensors="pt")[0]
            new_ids = new_ids.to(device=device, dtype=ids.dtype)
            
            # 5. 创建对应的 attention mask
            new_mask = torch.ones(len(new_ids), dtype=mask.dtype, device=device)
            
            new_input_ids_list.append(new_ids)
            new_attention_mask_list.append(new_mask)
        
        # Padding
        max_len = max(ids.size(0) for ids in new_input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        padded_input_ids = torch.full(
            (batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=device
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=device
        )
        
        for b, (ids, mask) in enumerate(zip(new_input_ids_list, new_attention_mask_list)):
            # 左侧 padding
            start = max_len - ids.size(0)
            padded_input_ids[b, start:] = ids
            padded_attention_mask[b, start:] = mask
        
        return padded_input_ids, padded_attention_mask
    
    def _mask_multimodal_inputs(
        self,
        multimodal_inputs: Dict[str, torch.Tensor],
        mask_ratio: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        遮蔽多模态输入（用于 Perception Counterfactual）
        
        策略：随机 mask 掉 mask_ratio 比例的值，保留少量信息
        - 避免全零导致音频处理模块崩溃
        - 同时大幅削弱多模态信息的影响
        
        Args:
            multimodal_inputs: 多模态输入字典
            mask_ratio: 遮蔽比例，默认 0.9（即保留 10%）
        """
        masked = {}
        
        # 需要遮蔽的关键词（音频/视频相关）
        mask_keywords = [
            "video", "audio", "pixel", "feature", "image",
            "input_features", "pixel_values", "image_features"
        ]
        
        # 不应该遮蔽的关键词（元信息）
        skip_keywords = [
            "attention_mask", "lengths", "position", "grid"
        ]
        
        for k, v in multimodal_inputs.items():
            k_lower = k.lower()
            
            # 检查是否应该跳过（元信息不能遮蔽）
            should_skip = any(kw in k_lower for kw in skip_keywords)
            if should_skip:
                masked[k] = v
                continue
            
            # 检查是否需要遮蔽
            should_mask = any(kw in k_lower for kw in mask_keywords)
            
            if should_mask and isinstance(v, torch.Tensor) and v.numel() > 0:
                # 创建随机 mask，mask_ratio 比例的位置为 0
                random_mask = (torch.rand_like(v.float()) > mask_ratio).to(v.dtype)
                masked[k] = v * random_mask
            else:
                masked[k] = v
        
        return masked
    
    def _get_answer_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multimodal_inputs: Dict[str, torch.Tensor],
        answer_start_positions: List[int],
        answer_end_positions: List[int],
    ) -> torch.Tensor:
        """获取 answer 部分的 logits"""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **multimodal_inputs
            )
            logits = outputs.logits
        
        batch_size = logits.size(0)
        max_answer_len = max(max(e - s, 1) for s, e in zip(answer_start_positions, answer_end_positions))
        vocab_size = logits.size(-1)
        device = logits.device
        
        answer_logits = torch.zeros(batch_size, max_answer_len, vocab_size, device=device)
        
        for b in range(batch_size):
            start = answer_start_positions[b]
            end = answer_end_positions[b]
            length = end - start
            
            if length > 0 and start >= 0:
                logits_start = max(0, start - 1)
                logits_end = end - 1
                actual_len = logits_end - logits_start
                
                if actual_len > 0:
                    answer_logits[b, :actual_len] = logits[b, logits_start:logits_end]
        
        return answer_logits
    
    def compute_rewards(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multimodal_inputs: Dict[str, torch.Tensor],
        gt_categories_list: List[List[str]],
        completion_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Logit-based Rewards
        
        执行三路 Forward:
        - Forward A: Full State
        - Forward B: Perception Counterfactual (遮蔽多模态)
        - Forward C: Coherence Counterfactual (移除推理)
        
        Args:
            model: 模型
            input_ids: (batch_size, seq_len) 完整的 prompt + completion ids
            attention_mask: (batch_size, seq_len) attention mask
            multimodal_inputs: 多模态输入
            gt_categories_list: 每个样本的 GT 情感词列表
            completion_ids: (batch_size, completion_len) completion 部分的 ids（可选）
        
        Returns:
            rewards: {
                "coherence": (batch_size,) coherence reward,
                "perception": (batch_size,) perception reward,
                "total": (batch_size,) 总 reward
            }
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 1. 找到每个样本的 answer 区域
        answer_start_positions = []
        answer_end_positions = []
        answer_token_ids_list = []
        
        for b in range(batch_size):
            start, end = self._extract_answer_region(input_ids[b])
            answer_start_positions.append(start)
            answer_end_positions.append(end)
            
            if start >= 0 and end > start:
                answer_token_ids_list.append(input_ids[b, start:end])
            else:
                answer_token_ids_list.append(torch.tensor([], dtype=input_ids.dtype, device=device))
        
        # 2. Forward A: Full State
        logits_A = self._get_answer_logits(
            model, input_ids, attention_mask, multimodal_inputs,
            answer_start_positions, answer_end_positions
        )
        
        # 初始化结果
        coherence_rewards = torch.zeros(batch_size, device=device)
        perception_rewards = torch.zeros(batch_size, device=device)
        scores_A = torch.zeros(batch_size, device=device)
        
        # 3. Forward C: Coherence Counterfactual (如果启用)
        logits_C = None
        if self.config.use_coherence_reward:
            try:
                no_think_ids, no_think_mask = self._build_no_think_input(input_ids, attention_mask)
                
                no_think_answer_starts = []
                no_think_answer_ends = []
                for b in range(batch_size):
                    start, end = self._extract_answer_region(no_think_ids[b])
                    no_think_answer_starts.append(start)
                    no_think_answer_ends.append(end)
                
                logits_C = self._get_answer_logits(
                    model, no_think_ids, no_think_mask, multimodal_inputs,
                    no_think_answer_starts, no_think_answer_ends
                )
            except Exception as e:
                print(f"[LogitReward WARNING] Coherence forward failed, skipping: {e}")
                logits_C = None
        
        # 4. Forward B: Perception Counterfactual (如果启用)
        logits_B = None
        if self.config.use_perception_reward:
            try:
                masked_mm_inputs = self._mask_multimodal_inputs(multimodal_inputs)
                
                logits_B = self._get_answer_logits(
                    model, input_ids, attention_mask, masked_mm_inputs,
                    answer_start_positions, answer_end_positions
                )
            except Exception as e:
                # 如果遮蔽多模态输入导致 forward 失败，跳过 perception reward
                print(f"[LogitReward WARNING] Perception forward failed, skipping: {e}")
                logits_B = None
        
        # 5. 计算每个样本的 reward（使用多 wheel 计算）
        for b in range(batch_size):
            answer_ids = answer_token_ids_list[b]
            gt_words = gt_categories_list[b]
            
            if len(answer_ids) == 0 or len(gt_words) == 0:
                continue
            
            actual_len = len(answer_ids)
            
            # 构建锚点 mask
            anchor_mask = self.token_map.build_anchor_mask(answer_ids)
            
            if anchor_mask.sum() == 0:
                continue
            
            # 提取该样本的 logits
            sample_logits_A = logits_A[b, :actual_len]  # Full State
            
            # 计算 coherence reward:
            # Gain(t) = [S(GT)_Full - S(GT)_NoThink] - [S(Neg)_Full - S(Neg)_NoThink]
            # r_coh = mean_t(Gain(t))
            if self.config.use_coherence_reward and logits_C is not None:
                sample_logits_C = logits_C[b, :actual_len]  # No-Think
                
                # 使用 compute_sequence_reward 按公式计算
                # 这里 logits_A 是 with-think, logits_C 是 no-think
                coherence_reward = self._compute_sequence_gain(
                    sample_logits_A, sample_logits_C, anchor_mask, gt_words
                )
                coherence_rewards[b] = coherence_reward
            
            # 计算 perception reward:
            # Gain(t) = [S(GT)_Full - S(GT)_NoMM] - [S(Neg)_Full - S(Neg)_NoMM]
            # r_prcp = mean_t(Gain(t))
            if self.config.use_perception_reward and logits_B is not None:
                sample_logits_B = logits_B[b, :actual_len]  # No-MM
                
                perception_reward = self._compute_sequence_gain(
                    sample_logits_A, sample_logits_B, anchor_mask, gt_words
                )
                perception_rewards[b] = perception_reward
            
            # 记录 Full State 得分
            scores_A[b] = self._compute_sample_gt_score(sample_logits_A, anchor_mask, gt_words)
        
        # 6. 归一化（可选）
        if self.config.normalize_rewards:
            if coherence_rewards.std() > 1e-6:
                coherence_rewards = (coherence_rewards - coherence_rewards.mean()) / (coherence_rewards.std() + 1e-6)
            if perception_rewards.std() > 1e-6:
                perception_rewards = (perception_rewards - perception_rewards.mean()) / (perception_rewards.std() + 1e-6)
        
        # 7. 计算总 reward
        total_rewards = (
            self.config.alpha * coherence_rewards +
            self.config.beta * perception_rewards
        )
        
        return {
            "coherence": coherence_rewards,
            "perception": perception_rewards,
            "total": total_rewards,
            "score_full": scores_A,
        }
    
    def _compute_sequence_gain(
        self,
        logits_with: torch.Tensor,
        logits_no: torch.Tensor,
        anchor_mask: torch.Tensor,
        gt_words: List[str],
    ) -> torch.Tensor:
        """
        计算序列级别的增益差分
        
        公式:
        Gain(t) = [S(GT, t)_with - S(GT, t)_no] - [S(Neg, t)_with - S(Neg, t)_no]
        R = (1 / Σ M_t) * Σ_t (M_t · Gain(t))
        
        Args:
            logits_with: (seq_len, vocab_size) With 条件的 logits
            logits_no: (seq_len, vocab_size) Without 条件的 logits
            anchor_mask: (seq_len,) 锚点 mask
            gt_words: GT 情感词列表
        
        Returns:
            reward: scalar
        """
        device = logits_with.device
        seq_len = logits_with.size(0)
        
        if anchor_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        gains = torch.zeros(seq_len, device=device)
        
        for t in range(seq_len):
            if anchor_mask[t] == 0:
                continue
            
            # 在该位置计算多 wheel 增益差分
            gain = self.scorer.compute_multi_wheel_gain(
                logits_with[t:t+1],  # (1, vocab_size)
                logits_no[t:t+1],
                gt_words
            )
            gains[t] = gain.squeeze()
        
        # 加权平均
        masked_gains = gains * anchor_mask.float()
        mask_sum = anchor_mask.float().sum() + 1e-10
        reward = masked_gains.sum() / mask_sum
        
        return reward
    
    def _compute_sample_gt_score(
        self,
        logits: torch.Tensor,
        anchor_mask: torch.Tensor,
        gt_words: List[str],
    ) -> torch.Tensor:
        """
        计算单个样本在锚点位置的 GT 置信度得分
        
        使用多 wheel 计算方式
        
        Args:
            logits: (seq_len, vocab_size)
            anchor_mask: (seq_len,)
            gt_words: GT 情感词列表
        
        Returns:
            score: scalar
        """
        device = logits.device
        
        if anchor_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # 找到 GT 在哪些 wheel 中
        wheel_to_l1 = self.token_map.find_gt_in_wheels(gt_words)
        
        if not wheel_to_l1:
            return torch.tensor(0.0, device=device)
        
        # 在每个锚点位置计算 GT 类别的平均置信度
        anchor_positions = anchor_mask.nonzero(as_tuple=True)[0]
        position_scores = []
        
        for t in anchor_positions:
            pos_logits = logits[t:t+1]  # (1, vocab_size)
            
            wheel_scores = []
            for wheel_name, gt_l1_set in wheel_to_l1.items():
                for l1 in gt_l1_set:
                    score = self.scorer.compute_wheel_l1_score(pos_logits, wheel_name, l1)
                    wheel_scores.append(score)
            
            if wheel_scores:
                avg_score = torch.stack(wheel_scores).mean()
                position_scores.append(avg_score)
        
        if position_scores:
            return torch.stack(position_scores).mean()
        else:
            return torch.tensor(0.0, device=device)


def create_logit_reward_computer(
    tokenizer,
    emotion_wheel_root: Optional[str] = None,
    config: Optional[LogitRewardConfig] = None,
) -> LogitRewardComputer:
    """
    工厂函数：创建 LogitRewardComputer
    
    Args:
        tokenizer: HuggingFace tokenizer
        emotion_wheel_root: 情绪轮数据路径（可选，有默认值）
        config: 配置对象
    
    Returns:
        LogitRewardComputer 实例
    """
    config = config or LogitRewardConfig()
    
    # 确定缓存路径
    cache_path = config.cache_path
    if cache_path is None:
        # 使用默认缓存路径
        default_cache = os.path.join(
            os.path.dirname(__file__),
            "emotion_token_map_cache.json"
        )
        if os.path.exists(default_cache):
            cache_path = default_cache
    
    # 创建 Token 映射表
    token_map = create_emotion_token_map(
        tokenizer=tokenizer,
        emotion_wheel_root=emotion_wheel_root,
        mapping_depth=config.mapping_depth,
        cache_path=cache_path,
    )
    
    return LogitRewardComputer(
        token_map=token_map,
        config=config,
        tokenizer=tokenizer,
    )
