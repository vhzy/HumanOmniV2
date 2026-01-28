#!/usr/bin/env python3
"""
根据pope_audio_filtered.jsonl过滤多个结果文件，只保留存在的记录
"""
import json
import os

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

# 读取pope_audio_filtered.jsonl作为基准
print("=" * 80)
print("【读取基准文件】")
print("=" * 80)
base_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered.jsonl"
base_data = load_jsonl(base_file)
print(f"基准文件: {base_file}")
print(f"记录数: {len(base_data)}")

# 创建基准键集合（使用name和question的组合）
base_keys = set()
for item in base_data:
    key = (item['name'], item['question'])
    base_keys.add(key)

print(f"唯一键数量: {len(base_keys)}\n")

# 要处理的文件列表
result_files = [
    "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo23_v2/pope_results_a2.jsonl",
    "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/pope_results_a2.jsonl",
    "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_baseline_2/pope_results_a2.jsonl",
    "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B/pope_results_a2.jsonl",
]

print("=" * 80)
print("【处理结果文件】")
print("=" * 80)

for result_file in result_files:
    print(f"\n处理文件: {result_file}")
    
    # 检查文件是否存在
    if not os.path.exists(result_file):
        print(f"  ⚠️  文件不存在，跳过")
        continue
    
    # 读取结果文件
    result_data = load_jsonl(result_file)
    print(f"  原始记录数: {len(result_data)}")
    
    # 过滤只保留在基准文件中的记录
    filtered_data = []
    removed_count = 0
    
    for item in result_data:
        key = (item['name'], item['question'])
        if key in base_keys:
            filtered_data.append(item)
        else:
            removed_count += 1
    
    print(f"  保留记录数: {len(filtered_data)}")
    print(f"  删除记录数: {removed_count}")
    
    # 生成新文件名（把a2改成a22）
    output_file = result_file.replace('pope_results_a2.jsonl', 'pope_results_a22.jsonl')
    
    # 保存过滤后的数据
    save_jsonl(filtered_data, output_file)
    print(f"  ✓ 已保存到: {output_file}")

print("\n" + "=" * 80)
print("【完成】")
print("=" * 80)
print("所有文件处理完成！")

