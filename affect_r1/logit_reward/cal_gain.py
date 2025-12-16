from emotion_token_map import create_emotion_token_map
from logit_scorer_v2 import create_logit_scorer

# 加载（首次构建，后续从缓存加载）
token_map = create_emotion_token_map(
    tokenizer=tokenizer,
    cache_path="emotion_token_map_cache.json"
)

# 创建打分器
scorer = create_logit_scorer(token_map=token_map)

# 计算增益（自动按 GT 所属的 wheel 分别计算后平均）
gain = scorer.compute_multi_wheel_gain(logits_with, logits_no, gt_words)