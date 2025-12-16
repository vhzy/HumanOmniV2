#!/usr/bin/env python3
"""
根据索引提取实际的数据内容
"""
import json

# 读取索引信息
indices_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data_indices_step_2300_2620.json"
with open(indices_file, 'r', encoding='utf-8') as f:
    indices_data = json.load(f)

# 读取原始数据
data_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_with_extracted_clues_fixed.jsonl"
all_data = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line.strip()))

print(f"总数据量: {len(all_data)}")

# 获取unique索引
unique_indices = indices_data["unique_data_indices"]
steps_detail = indices_data["steps_detail"]

print(f"需要提取的unique数据数量: {len(unique_indices)}")

# 提取对应的数据
extracted_data = []
for idx in unique_indices:
    if idx < len(all_data):
        data_item = all_data[idx].copy()
        data_item["original_index"] = idx
        extracted_data.append(data_item)

# 按step组织数据
data_by_step = {}
for step_str, step_info in steps_detail.items():
    step = int(step_str)
    step_data = []
    for idx in step_info["unique_indices"]:
        if idx < len(all_data):
            data_item = all_data[idx].copy()
            data_item["original_index"] = idx
            step_data.append(data_item)
    data_by_step[step] = step_data

# 保存提取的数据
output_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/extracted_data_step_2300_2620.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in extracted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存按step组织的数据
output_by_step = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data_by_step_2300_2620.json"
with open(output_by_step, 'w', encoding='utf-8') as f:
    json.dump(data_by_step, f, indent=2, ensure_ascii=False)

print(f"\n已提取 {len(extracted_data)} 条unique数据")
print(f"保存到:")
print(f"  - {output_file} (所有unique数据, JSONL格式)")
print(f"  - {output_by_step} (按step组织的数据, JSON格式)")

# 打印一些统计信息
print(f"\n每个step的数据样本示例:")
print("-" * 80)
for step in [2300, 2301, 2302, 2619, 2620]:
    if step in data_by_step:
        print(f"Step {step}: {len(data_by_step[step])} 个样本")
        if data_by_step[step]:
            sample = data_by_step[step][0]
            print(f"  第一个样本: index={sample['original_index']}")
            if 'path' in sample:
                print(f"    path: {sample['path']}")
            if 'problem' in sample:
                print(f"    problem: {sample['problem'][:100]}..." if len(sample['problem']) > 100 else f"    problem: {sample['problem']}")
        print()

