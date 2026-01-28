#!/usr/bin/env python3
"""
数据处理脚本：
1. 从文件1中提取不在文件2中的数据（差集）
2. 加上文件3的数据
3. 如果不足1万条，从文件1随机选取补充
4. 输出两个文件：
   - 1万条数据（去除extracted_clues）
   - 剩余的文件1数据（保留所有字段）
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set

def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存JSONL文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条数据到: {filepath}")

def get_unique_key(item: Dict) -> str:
    """生成数据的唯一标识"""
    # 使用path或video字段作为唯一标识
    return item.get('path') or item.get('video', '')

def remove_field(data: List[Dict], field: str) -> List[Dict]:
    """移除指定字段"""
    result = []
    for item in data:
        new_item = {k: v for k, v in item.items() if k != field}
        result.append(new_item)
    return result

def main():
    # 文件路径
    base_dir = Path('/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data')
    file1_path = base_dir / 'rl_with_extracted_clues_fixed.jsonl'
    file2_path = base_dir / 'rl_with_extracted_clues_filtered.jsonl'
    file3_path = base_dir / 'rl_data.jsonl'
    
    output_10k_path = base_dir / 'rl_10k_no_clues.jsonl'
    output_remaining_path = base_dir / 'rl_remaining_with_clues.jsonl'
    
    print("=" * 60)
    print("开始处理数据...")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据文件...")
    data1 = load_jsonl(file1_path)
    data2 = load_jsonl(file2_path)
    data3 = load_jsonl(file3_path)
    
    print(f"  - 文件1 (fixed): {len(data1)} 条")
    print(f"  - 文件2 (filtered): {len(data2)} 条")
    print(f"  - 文件3 (rl_data): {len(data3)} 条")
    
    # 2. 创建标识集合
    print("\n[2/5] 计算差集和重合...")
    keys1 = {get_unique_key(item): item for item in data1}
    keys2 = {get_unique_key(item) for item in data2}
    keys3 = {get_unique_key(item) for item in data3}
    
    # 差集：文件1 - 文件2
    diff_keys = set(keys1.keys()) - keys2
    diff_data = [keys1[key] for key in diff_keys]
    print(f"  - 差集 (文件1 - 文件2): {len(diff_data)} 条")
    
    # 3. 构建1万条数据集
    print("\n[3/5] 构建1万条数据集...")
    selected_data = []
    used_keys = set()
    
    # 首先添加差集数据
    selected_data.extend(diff_data)
    used_keys.update(diff_keys)
    print(f"  - 添加差集数据: {len(diff_data)} 条")
    
    # 添加文件3数据（避免重复）
    added_from_file3 = 0
    for item in data3:
        key = get_unique_key(item)
        if key not in used_keys:
            selected_data.append(item)
            used_keys.add(key)
            added_from_file3 += 1
        # 如果文件3中的数据已经在used_keys中，也添加（因为用户说有重合）
        elif key in keys3:
            selected_data.append(item)
            added_from_file3 += 1
    
    print(f"  - 添加文件3数据: {added_from_file3} 条")
    print(f"  - 当前总数: {len(selected_data)} 条")
    
    # 如果不足1万条，从文件1随机选取补充
    target_count = 10000
    if len(selected_data) < target_count:
        needed = target_count - len(selected_data)
        available_keys = [k for k in keys1.keys() if k not in used_keys]
        
        if len(available_keys) >= needed:
            random.seed(42)  # 设置随机种子以保证可重复
            sampled_keys = random.sample(available_keys, needed)
            sampled_data = [keys1[key] for key in sampled_keys]
            selected_data.extend(sampled_data)
            used_keys.update(sampled_keys)
            print(f"  - 随机补充文件1数据: {needed} 条")
        else:
            print(f"  ⚠️ 警告: 可用数据不足，只能添加 {len(available_keys)} 条")
            sampled_data = [keys1[key] for key in available_keys]
            selected_data.extend(sampled_data)
            used_keys.update(available_keys)
    
    print(f"  - 最终选中数据: {len(selected_data)} 条")
    
    # 4. 移除extracted_clues字段并保存
    print("\n[4/5] 移除extracted_clues字段并保存1万条数据...")
    selected_data_no_clues = remove_field(selected_data, 'extracted_clues')
    save_jsonl(selected_data_no_clues, output_10k_path)
    
    # 5. 保存剩余的文件1数据
    print("\n[5/5] 保存剩余的文件1数据（保留所有字段）...")
    remaining_keys = set(keys1.keys()) - used_keys
    remaining_data = [keys1[key] for key in remaining_keys]
    print(f"  - 剩余数据: {len(remaining_data)} 条")
    save_jsonl(remaining_data, output_remaining_path)
    
    # 统计总结
    print("\n" + "=" * 60)
    print("处理完成！统计信息：")
    print("=" * 60)
    print(f"原始文件1数据: {len(data1)} 条")
    print(f"已使用数据: {len(used_keys)} 条")
    print(f"  - 差集数据: {len(diff_data)} 条")
    print(f"  - 文件3数据: {added_from_file3} 条")
    print(f"  - 随机补充: {len(selected_data) - len(diff_data) - added_from_file3} 条")
    print(f"剩余数据: {len(remaining_data)} 条")
    print(f"\n输出文件:")
    print(f"  1. {output_10k_path.name} ({len(selected_data_no_clues)} 条，无extracted_clues)")
    print(f"  2. {output_remaining_path.name} ({len(remaining_data)} 条，保留所有字段)")
    print("=" * 60)

if __name__ == '__main__':
    main()
