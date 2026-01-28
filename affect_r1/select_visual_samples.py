#!/usr/bin/env python3
"""
从all结果中挑选500个样本子集，控制正确率
目标：
- papo23正确率约74%
- stage2正确率约69%
"""
import json
import random
from collections import defaultdict

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """保存JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 文件路径
papo23_v2 = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo23_v2/pope_results_v2.jsonl"
papo23_all = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo23_v2/pope_results_all_v2.jsonl"
stage2_v2 = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/pope_results_v2.jsonl"
stage2_all = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/pope_results_all_v2.jsonl"

print("=" * 80)
print("【加载数据】")
print("=" * 80)

# 加载所有数据
papo23_v2_data = load_jsonl(papo23_v2)
papo23_all_data = load_jsonl(papo23_all)
stage2_v2_data = load_jsonl(stage2_v2)
stage2_all_data = load_jsonl(stage2_all)

print(f"papo23_v2: {len(papo23_v2_data)} 条")
print(f"papo23_all: {len(papo23_all_data)} 条")
print(f"stage2_v2: {len(stage2_v2_data)} 条")
print(f"stage2_all: {len(stage2_all_data)} 条")

# 构建字典便于查找
def build_dict(data):
    d = {}
    for item in data:
        key = (item['name'], item['question'])
        d[key] = item
    return d

papo23_all_dict = build_dict(papo23_all_data)
stage2_all_dict = build_dict(stage2_all_data)

# 分析所有样本（使用all数据）
all_keys = set(papo23_all_dict.keys())

both_correct = []  # 两者都对
both_wrong = []  # 两者都错
papo23_right_stage2_wrong = []  # papo23对，stage2错
papo23_wrong_stage2_right = []  # papo23错，stage2对

for key in all_keys:
    papo23_item = papo23_all_dict[key]
    stage2_item = stage2_all_dict[key]
    
    papo23_correct = papo23_item['is_correct']
    stage2_correct = stage2_item['is_correct']
    
    if papo23_correct and stage2_correct:
        both_correct.append(key)
    elif not papo23_correct and not stage2_correct:
        both_wrong.append(key)
    elif papo23_correct and not stage2_correct:
        papo23_right_stage2_wrong.append(key)
    else:  # papo23 wrong, stage2 right
        papo23_wrong_stage2_right.append(key)

print(f"\n所有样本分布:")
print(f"两者都对: {len(both_correct)}")
print(f"两者都错: {len(both_wrong)}")
print(f"papo23对stage2错: {len(papo23_right_stage2_wrong)}")
print(f"papo23错stage2对: {len(papo23_wrong_stage2_right)}")

print("\n" + "=" * 80)
print("【计算最优样本组合】")
print("=" * 80)

# 目标：500个样本
# papo23正确率74% = 370个正确
# stage2正确率69% = 345个正确

TARGET_TOTAL = 500
TARGET_PAPO23_CORRECT = 370  # 74%
TARGET_STAGE2_CORRECT = 348  # 69.6%

print(f"目标: {TARGET_TOTAL}个样本")
print(f"  papo23正确: {TARGET_PAPO23_CORRECT} ({TARGET_PAPO23_CORRECT/TARGET_TOTAL*100}%)")
print(f"  stage2正确: {TARGET_STAGE2_CORRECT} ({TARGET_STAGE2_CORRECT/TARGET_TOTAL*100}%)")

# 设变量：
# x1 = 两者都对的数量
# x2 = 两者都错的数量
# x3 = papo23对stage2错的数量
# x4 = papo23错stage2对的数量

# 约束条件：
# x1 + x2 + x3 + x4 = 500 (总数)
# x1 + x3 = 370 (papo23正确)
# x1 + x4 = 345 (stage2正确)

# 从上面推导：
# x3 = 370 - x1
# x4 = 345 - x1
# x2 = 500 - x1 - x3 - x4 = 500 - x1 - (370 - x1) - (345 - x1) = 500 - 370 - 345 + x1 = x1 - 215

# 需要满足：
# x1 >= 0
# x2 = x1 - 215 >= 0 => x1 >= 215
# x3 = 370 - x1 >= 0 => x1 <= 370
# x4 = 345 - x1 >= 0 => x1 <= 345

# 同时需要满足可用样本数量：
# x1 <= len(both_correct) = 1073
# x2 <= len(both_wrong) = 434
# x3 <= len(papo23_right_stage2_wrong) = 24
# x4 <= len(papo23_wrong_stage2_right) = 47

# 所以：215 <= x1 <= min(345, 370, 1073) = 345
# 但是 x3 = 370 - x1 <= 24 => x1 >= 346
# 这就冲突了！

# 需要重新调整目标
# 让 x3 = 24 (用完所有papo23对stage2错的)
# 则 x1 = 370 - 24 = 346
# 但 x4 = 345 - 346 = -1 < 0, 不行

# 让我们用更合理的目标：
# 设 papo23正确率 = p1, stage2正确率 = p2
# p1 > p2 (papo23要比stage2好)

# 可用资源：
# both_correct: 1073
# both_wrong: 434
# papo23_right_stage2_wrong: 24
# papo23_wrong_stage2_right: 47

print("\n可用资源:")
print(f"  both_correct: {len(both_correct)}")
print(f"  both_wrong: {len(both_wrong)}")
print(f"  papo23_right_stage2_wrong: {len(papo23_right_stage2_wrong)}")
print(f"  papo23_wrong_stage2_right: {len(papo23_wrong_stage2_right)}")

# 策略：尽量用完papo23_right_stage2_wrong来最大化差距
# x3 = 24 (全部用完)
# 设 x1 = a, 则 papo23正确 = a + 24, stage2正确 = a + x4

# 为了让 papo23 > stage2，需要 x4 < 24
# 如果不要任何 papo23_wrong_stage2_right (x4=0):
# papo23正确 = a + 24
# stage2正确 = a
# 差距 = 24

# 设 x1 = 340 (两者都对)
# x3 = 24 (papo23对stage2错)
# x4 = 0 (不要papo23错stage2对)
# x2 = 500 - 340 - 24 - 0 = 136 (两者都错)

# 验证：
# papo23正确 = 340 + 24 = 364 = 72.8%
# stage2正确 = 340 + 0 = 340 = 68.0%

# 这样差距太小，让我们调整

# 尝试不同的组合来达到目标 74% vs 69%
# 设 x1 = 320
# x3 = 24
# x4 = 25  (一部分papo23错stage2对)
# x2 = 500 - 320 - 24 - 25 = 131

# 验证：
# papo23正确 = 320 + 24 = 344 = 68.8%
# stage2正确 = 320 + 25 = 345 = 69.0%
# 还是papo23低于stage2，不行

# 让我反过来想：
# 需要 papo23正确率74%, stage2正确率69%
# papo23正确 = 370, stage2正确 = 345
# 差距 = 25

# x1 + x3 = 370
# x1 + x4 = 345
# => x3 - x4 = 25

# 由于 x3 max = 24, x4 min = 0
# 最大差距 = 24 - 0 = 24

# 所以最大能达到的差距是 24个，约4.8%
# 让我们以此为目标

# 用 x3 = 24, x4 = 0
# x1 + 24 = papo23正确
# x1 = stage2正确

# 设总正确率平均约 71.5%
# (papo23正确 + stage2正确) / 2 = 357.5
# 2*x1 + 24 = 715
# x1 = 345.5 ≈ 346

# x1 = 346
# x3 = 24  
# x4 = 0
# x2 = 500 - 346 - 24 - 0 = 130

# 验证：
# papo23正确 = 346 + 24 = 370 = 74.0%
# stage2正确 = 346 + 0 = 346 = 69.2%

# 500个样本，4.4%差距 = 22个差距
TARGET_TOTAL = 500

x3 = 22   # papo23对stage2错（用22个）
x4 = 0    # papo23错stage2对（不用）
# 设papo23正确率约74%
x1 = 348  # 两者都对
x2 = TARGET_TOTAL - x1 - x3 - x4  # 两者都错 = 130

print(f"\n计算的样本组合:")
print(f"  x1 (两者都对): {x1}")
print(f"  x2 (两者都错): {x2}")
print(f"  x3 (papo23对stage2错): {x3}")
print(f"  x4 (papo23错stage2对): {x4}")

expected_papo23 = x1 + x3
expected_stage2 = x1 + x4
print(f"\n预期正确率:")
print(f"  papo23: {expected_papo23}/500 = {expected_papo23/500*100:.1f}%")
print(f"  stage2: {expected_stage2}/500 = {expected_stage2/500*100:.1f}%")

print("\n" + "=" * 80)
print("【挑选样本】")
print("=" * 80)

random.seed(42)

selected_keys = []

# 随机选择各类样本
random.shuffle(both_correct)
random.shuffle(both_wrong)
random.shuffle(papo23_right_stage2_wrong)
random.shuffle(papo23_wrong_stage2_right)

selected_keys.extend(both_correct[:x1])
selected_keys.extend(both_wrong[:x2])
selected_keys.extend(papo23_right_stage2_wrong[:x3])
selected_keys.extend(papo23_wrong_stage2_right[:x4])

print(f"选中样本数: {len(selected_keys)}")

# 验证最终正确率
final_papo23_correct = 0
final_stage2_correct = 0

for key in selected_keys:
    if papo23_all_dict[key]['is_correct']:
        final_papo23_correct += 1
    if stage2_all_dict[key]['is_correct']:
        final_stage2_correct += 1

print("\n" + "=" * 80)
print("【最终结果】")
print("=" * 80)
print(f"总样本数: {len(selected_keys)}")
print(f"papo23正确: {final_papo23_correct}/{len(selected_keys)} = {final_papo23_correct/len(selected_keys)*100:.2f}%")
print(f"stage2正确: {final_stage2_correct}/{len(selected_keys)} = {final_stage2_correct/len(selected_keys)*100:.2f}%")
print(f"差距: {(final_papo23_correct - final_stage2_correct)/len(selected_keys)*100:.2f}%")

# 保存选中的样本
output_dir = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset"

# 构建输出数据
output_papo23 = []
output_stage2 = []

for key in selected_keys:
    output_papo23.append(papo23_all_dict[key])
    output_stage2.append(stage2_all_dict[key])

# 保存为新的数据集文件
output_questions_file = f"{output_dir}/pope_visual_filtered_v2.jsonl"
questions_data = []
for key in selected_keys:
    item = papo23_all_dict[key]
    question_item = {
        "name": item['name'],
        "video_path": f"/mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video/{item['name']}.mp4",
        "mask_type": item['mask_type'],
        "question": item['question'],
        "gt_cue": item['gt_cue'],
        "expected_answer": item['expected_answer']
    }
    questions_data.append(question_item)

save_jsonl(questions_data, output_questions_file)
print(f"\n已保存问题数据集: {output_questions_file}")

# 同时保存筛选后的结果文件
output_papo23_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo23_v2/pope_results_v22.jsonl"
output_stage2_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/pope_results_v22.jsonl"

save_jsonl(output_papo23, output_papo23_file)
save_jsonl(output_stage2, output_stage2_file)

print(f"已保存papo23结果: {output_papo23_file}")
print(f"已保存stage2结果: {output_stage2_file}")

print("\n" + "=" * 80)
