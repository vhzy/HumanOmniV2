"""
Logit Reward 模块

提供基于 Logits 的情感推理 Reward 计算，包括:
- Coherence Reward: 推理过程对答案的信息增益
- Perception Reward: 视觉/音频输入对答案的信息增益

核心设计:
- 使用多 wheel 计算：GT 映射到 n 个情绪轮，分别计算后平均
- 每个 wheel 内部独立检测 Token 冲突，保证 Level1 类别之间不歧义

Usage:
    from affect_r1.logit_reward import (
        LogitRewardComputer,
        LogitRewardConfig,
        create_logit_reward_computer,
    )
    
    # 创建计算器
    config = LogitRewardConfig(alpha=0.1, beta=0.1)
    computer = create_logit_reward_computer(tokenizer, config=config)
    
    # 计算 reward
    rewards = computer.compute_rewards(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multimodal_inputs=multimodal_inputs,
        gt_categories_list=gt_categories_list,
    )
"""

# 新版本模块（推荐使用）
from .emotion_token_map import (
    EmotionTokenMap,
    EmotionTokenMapBuilder,
    create_emotion_token_map,
)
from .logit_scorer_v2 import (
    LogitScorerV2,
    create_logit_scorer,
)
from .reward_computer import (
    LogitRewardConfig,
    LogitRewardComputer,
    create_logit_reward_computer,
)

__all__ = [
    # 核心接口
    "LogitRewardConfig",
    "LogitRewardComputer",
    "create_logit_reward_computer",
    # Token 映射
    "EmotionTokenMap",
    "EmotionTokenMapBuilder",
    "create_emotion_token_map",
    # Logit 打分
    "LogitScorerV2",
    "create_logit_scorer",
]
