#!/usr/bin/env python3
"""
重新处理数据，确保无重复：
1. 优先保留文件3的数据
2. 添加差集数据（文件1 - 文件2）
3. 从remaining文件补充到1万条
4. 确保所有数据无重复
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
    remaining_path = base_dir / 'rl_remaining_with_clues.jsonl'
    
    output_10k_path = base_dir / 'rl_10k_no_clues.jsonl'
    backup_path = base_dir / 'rl_10k_no_clues_backup.jsonl'
    
    print("=" * 70)
    print("重新处理数据（优先保留文件3，从remaining补充）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据文件...")
    data1 = load_jsonl(file1_path)
    data2 = load_jsonl(file2_path)
    data3 = load_jsonl(file3_path)
    remaining_data = load_jsonl(remaining_path)
    
    print(f"  - 文件1 (fixed): {len(data1)} 条")
    print(f"  - 文件2 (filtered): {len(data2)} 条")
    print(f"  - 文件3 (rl_data) [优先级最高]: {len(data3)} 条")
    print(f"  - remaining文件: {len(remaining_data)} 条")
    
    # 2. 创建索引
    print("\n[2/6] 创建数据索引...")
    keys1 = {get_unique_key(item): item for item in data1}
    keys2 = {get_unique_key(item) for item in data2}
    keys3_dict = {get_unique_key(item): item for item in data3}
    remaining_dict = {get_unique_key(item): item for item in remaining_data}
    
    # 3. 计算差集
    print("\n[3/6] 计算差集...")
    diff_keys = set(keys1.keys()) - keys2
    print(f"  - 差集 (文件1 - 文件2): {len(diff_keys)} 条")
    
    # 4. 按优先级构建10k数据集
    print("\n[4/6] 按优先级构建10k数据集...")
    selected_data = []
    used_keys = set()
    
    # 优先级1: 文件3的数据
    print(f"  [优先级1] 添加文件3数据...")
    for key, item in keys3_dict.items():
        if key not in used_keys:
            selected_data.append(item)
            used_keys.add(key)
    print(f"    ✓ 添加了 {len(selected_data)} 条文件3数据")
    
    # 优先级2: 差集数据（文件1-文件2，但排除已有的）
    print(f"  [优先级2] 添加差集数据...")
    initial_count = len(selected_data)
    for key in diff_keys:
        if key not in used_keys:
            selected_data.append(keys1[key])
            used_keys.add(key)
    diff_added = len(selected_data) - initial_count
    print(f"    ✓ 添加了 {diff_added} 条差集数据")
    print(f"    当前总数: {len(selected_data)} 条")
    
    # 优先级3: 从remaining文件补充
    target_count = 10000
    if len(selected_data) < target_count:
        needed = target_count - len(selected_data)
        print(f"  [优先级3] 从remaining文件补充 {needed} 条数据...")
        
        # 获取remaining中不重复的数据
        available_keys = [k for k in remaining_dict.keys() if k not in used_keys]
        
        if len(available_keys) >= needed:
            random.seed(42)  # 设置随机种子以保证可重复
            sampled_keys = random.sample(available_keys, needed)
            for key in sampled_keys:
                selected_data.append(remaining_dict[key])
                used_keys.add(key)
            print(f"    ✓ 成功补充了 {needed} 条数据")
        else:
            print(f"    ⚠️ 警告: remaining文件可用数据不足！")
            print(f"    只能补充 {len(available_keys)} 条数据")
            for key in available_keys:
                selected_data.append(remaining_dict[key])
                used_keys.add(key)
    
    print(f"\n  最终数据总数: {len(selected_data)} 条")
    
    # 5. 验证无重复
    print("\n[5/6] 验证数据唯一性...")
    final_keys = [get_unique_key(item) for item in selected_data]
    unique_keys = set(final_keys)
    
    if len(final_keys) == len(unique_keys):
        print(f"  ✅ 验证通过：所有 {len(selected_data)} 条数据都是唯一的！")
    else:
        print(f"  ❌ 错误：发现重复数据！")
        print(f"  总数据: {len(final_keys)}, 唯一数据: {len(unique_keys)}")
        print(f"  重复数量: {len(final_keys) - len(unique_keys)}")
        return
    
    # 6. 备份旧文件并保存新文件
    print("\n[6/6] 保存文件...")
    
    # 备份旧文件
    if output_10k_path.exists():
        import shutil
        shutil.copy(output_10k_path, backup_path)
        print(f"  ✓ 已备份旧文件到: {backup_path.name}")
    
    # 移除extracted_clues字段
    selected_data_no_clues = remove_field(selected_data, 'extracted_clues')
    
    # 保存新文件
    save_jsonl(selected_data_no_clues, output_10k_path)
    
    # 统计总结
    print("\n" + "=" * 70)
    print("处理完成！数据来源统计：")
    print("=" * 70)
    
    # 统计每个来源的数据量
    file3_count = sum(1 for key in used_keys if key in keys3_dict)
    diff_count = sum(1 for key in used_keys if key in diff_keys and key not in keys3_dict)
    remaining_count = sum(1 for key in used_keys if key in remaining_dict and key not in keys3_dict and key not in diff_keys)
    
    print(f"数据来源分布:")
    print(f"  - 文件3数据（优先级最高）: {file3_count} 条")
    print(f"  - 差集数据（文件1-文件2）: {diff_count} 条")
    print(f"  - Remaining文件补充: {remaining_count} 条")
    print(f"  - 总计: {len(selected_data)} 条")
    print(f"\n输出文件:")
    print(f"  ✓ {output_10k_path.name} ({len(selected_data_no_clues)} 条，无extracted_clues，无重复)")
    if backup_path.exists():
        print(f"  ℹ️ 备份文件: {backup_path.name}")
    print("=" * 70)

if __name__ == '__main__':
    main()
