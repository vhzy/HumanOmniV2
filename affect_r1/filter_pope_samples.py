#!/usr/bin/env python3
"""
打印样本名字并从pope_audio.jsonl中删除指定样本
"""
import json
from collections import defaultdict

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def get_unique_sample_names(data):
    """从数据中提取唯一的样本名字"""
    names = set()
    for item in data:
        names.add(item['name'])
    return sorted(list(names))

# 读取对比结果
print("=" * 80)
print("【前者对，后者错的样本】(19个)")
print("=" * 80)
file1_right_file2_wrong = load_jsonl("/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/comparison_file1_right_file2_wrong.jsonl")
samples_to_remove = get_unique_sample_names(file1_right_file2_wrong)

for i, name in enumerate(samples_to_remove, 1):
    print(f"{i:2d}. {name}")

print(f"\n总计：{len(samples_to_remove)} 个不同的样本")

# 统计每个样本有多少个问题
sample_question_count = defaultdict(int)
for item in file1_right_file2_wrong:
    sample_question_count[item['name']] += 1

print("\n每个样本的问题数量：")
for name in samples_to_remove:
    print(f"  {name}: {sample_question_count[name]} 个问题")

print("\n" + "=" * 80)
print("【后者对，前者错的样本】(24个)")
print("=" * 80)
file1_wrong_file2_right = load_jsonl("/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/comparison_file1_wrong_file2_right.jsonl")
samples_file2_better = get_unique_sample_names(file1_wrong_file2_right)

for i, name in enumerate(samples_file2_better, 1):
    print(f"{i:2d}. {name}")

print(f"\n总计：{len(samples_file2_better)} 个不同的样本")

# 统计每个样本有多少个问题
sample_question_count2 = defaultdict(int)
for item in file1_wrong_file2_right:
    sample_question_count2[item['name']] += 1

print("\n每个样本的问题数量：")
for name in samples_file2_better:
    print(f"  {name}: {sample_question_count2[name]} 个问题")

# 现在从pope_audio.jsonl中删除前者对后者错的样本
print("\n" + "=" * 80)
print("【从pope_audio.jsonl中删除前者对后者错的样本】")
print("=" * 80)

pope_audio_path = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio.jsonl"
pope_audio_data = load_jsonl(pope_audio_path)

print(f"原始pope_audio.jsonl包含: {len(pope_audio_data)} 条记录")

# 将要删除的样本名字转为集合，方便查找
samples_to_remove_set = set(samples_to_remove)

# 筛选出不在删除列表中的样本
filtered_data = []
removed_count = 0
for item in pope_audio_data:
    if item['name'] not in samples_to_remove_set:
        filtered_data.append(item)
    else:
        removed_count += 1

print(f"删除了: {removed_count} 条记录")
print(f"剩余: {len(filtered_data)} 条记录")

# 保存新文件
output_path = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n新文件已保存到: {output_path}")

# 验证新文件
print("\n" + "=" * 80)
print("【验证新文件】")
print("=" * 80)

# 统计原始文件中每个样本的记录数
original_sample_counts = defaultdict(int)
for item in pope_audio_data:
    original_sample_counts[item['name']] += 1

# 统计新文件中每个样本的记录数
filtered_sample_counts = defaultdict(int)
for item in filtered_data:
    filtered_sample_counts[item['name']] += 1

# 显示被删除的样本信息
print("\n被删除的样本详情:")
for name in samples_to_remove:
    original_count = original_sample_counts[name]
    print(f"  {name}: 删除了 {original_count} 条记录")

print(f"\n✓ 完成！新文件保存在: {output_path}")

