import sys
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/logit_reward")

import torch
from transformers import AutoTokenizer
from reward_computer import create_logit_reward_computer, LogitRewardConfig

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
    local_files_only=True,
    trust_remote_code=True
)

config = LogitRewardConfig(
    use_coherence_reward=True,
    use_perception_reward=True,
    alpha=0.1,
    beta=0.1,
)
computer = create_logit_reward_computer(tokenizer=tokenizer, config=config)

print("="*70)
print("测试完整 Reward 计算流程")
print("="*70)

# 构建测试输入
full_text = "What emotion?<think>Happy face</think><answer>happy, joyful</answer>"
input_ids = torch.tensor([tokenizer.encode(full_text, add_special_tokens=False)])
attention_mask = torch.ones_like(input_ids)

print(f"\n输入序列长度: {input_ids.size(1)}")

# GT 类别
gt_categories_list = [["happy", "joyful"]]
print(f"GT 类别: {gt_categories_list}")

# 提取 answer 区域
start, end = computer._extract_answer_region(input_ids[0])
print(f"Answer 区域: [{start}, {end})")
print(f"Answer: '{tokenizer.decode(input_ids[0, start:end])}'")

# 构建锚点 mask
anchor_mask = computer.token_map.build_anchor_mask(input_ids[0, start:end])
print(f"锚点 Mask: {anchor_mask.tolist()}")

# 模拟 logits
vocab_size = len(tokenizer)
answer_len = end - start

logits_A = torch.randn(1, answer_len, vocab_size)  # Full State
logits_C = torch.randn(1, answer_len, vocab_size)  # No-Think

print("\n" + "="*70)
print("测试序列增益计算")
print("="*70)

# 测试 _compute_sequence_gain
sample_logits_A = logits_A[0]
sample_logits_C = logits_C[0]

coherence_gain = computer._compute_sequence_gain(
    sample_logits_A, sample_logits_C, anchor_mask, gt_categories_list[0]
)
print(f"\nCoherence 增益: {coherence_gain.item():.4f}")

# 测试 GT 得分
score_A = computer._compute_sample_gt_score(sample_logits_A, anchor_mask, gt_categories_list[0])
score_C = computer._compute_sample_gt_score(sample_logits_C, anchor_mask, gt_categories_list[0])
print(f"Score_A (Full): {score_A.item():.4f}")
print(f"Score_C (NoThink): {score_C.item():.4f}")

# 检查 GT 映射到哪些 wheel
wheel_to_l1 = computer.token_map.find_gt_in_wheels(gt_categories_list[0])
print(f"\nGT 映射:")
for wheel, l1s in sorted(wheel_to_l1.items()):
    print(f"  {wheel}: {l1s}")

print("\n" + "="*70)
print("✓ 测试完成")
print("="*70)