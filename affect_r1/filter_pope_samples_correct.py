#!/usr/bin/env python3
"""
只删除前者对后者错的19条具体问题记录
"""
import json

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# 读取前者对后者错的19条记录
print("=" * 80)
print("【前者对，后者错的样本】")
print("=" * 80)
file1_right_file2_wrong = load_jsonl("/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/comparison_file1_right_file2_wrong.jsonl")

print(f"总共有 {len(file1_right_file2_wrong)} 条要删除的记录\n")

# 打印要删除的记录
for i, item in enumerate(file1_right_file2_wrong, 1):
    print(f"{i:2d}. {item['name']}: {item['question'][:60]}...")

# 创建要删除的记录的键集合（使用name和question的组合作为唯一标识）
records_to_remove = set()
for item in file1_right_file2_wrong:
    key = (item['name'], item['question'])
    records_to_remove.add(key)

print(f"\n要删除的记录键数量: {len(records_to_remove)}")

# 读取pope_audio.jsonl
pope_audio_path = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio.jsonl"
pope_audio_data = load_jsonl(pope_audio_path)

print("\n" + "=" * 80)
print("【处理pope_audio.jsonl】")
print("=" * 80)
print(f"原始pope_audio.jsonl包含: {len(pope_audio_data)} 条记录")

# 只删除匹配的具体问题记录
filtered_data = []
removed_count = 0
removed_details = []

for item in pope_audio_data:
    key = (item['name'], item['question'])
    if key not in records_to_remove:
        filtered_data.append(item)
    else:
        removed_count += 1
        removed_details.append(f"{item['name']}: {item['question'][:60]}...")

print(f"删除了: {removed_count} 条记录")
print(f"剩余: {len(filtered_data)} 条记录")

# 显示被删除的记录
print("\n被删除的记录详情:")
for i, detail in enumerate(removed_details, 1):
    print(f"  {i:2d}. {detail}")

# 保存新文件
output_path = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✓ 完成！新文件保存在: {output_path}")
print(f"✓ 原始记录数: {len(pope_audio_data)}")
print(f"✓ 删除记录数: {removed_count}")
print(f"✓ 剩余记录数: {len(filtered_data)}")

