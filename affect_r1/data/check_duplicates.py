#!/usr/bin/env python3
"""检查JSONL文件中的重复样本"""

import json
from collections import Counter
from pathlib import Path

def check_duplicates(filepath: str):
    """检查文件中的重复样本"""
    print(f"检查文件: {filepath}")
    print("=" * 60)
    
    data = []
    keys = []
    
    # 读取数据
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append((idx, item))
                # 使用path或video作为唯一标识
                key = item.get('path') or item.get('video', '')
                keys.append(key)
    
    print(f"总样本数: {len(data)}")
    print(f"唯一键数: {len(set(keys))}")
    
    # 统计重复
    key_counts = Counter(keys)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    
    if duplicates:
        print(f"\n⚠️ 发现 {len(duplicates)} 个重复的样本键")
        print(f"重复样本总数: {sum(duplicates.values())}")
        print("\n重复详情:")
        print("-" * 60)
        
        for key, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"\n键: {key}")
            print(f"出现次数: {count}")
            
            # 显示这个键出现在哪些行
            lines = [idx for idx, item in data if (item.get('path') or item.get('video', '')) == key]
            print(f"行号: {lines}")
            
            # 显示第一个和最后一个样本的数据
            if len(lines) >= 2:
                first_item = data[lines[0]-1][1]
                last_item = data[lines[-1]-1][1]
                print(f"第一个样本数据: {json.dumps(first_item, ensure_ascii=False)[:200]}...")
                if first_item != last_item:
                    print(f"⚠️ 注意：相同键的样本数据不完全一致！")
                    print(f"最后一个样本数据: {json.dumps(last_item, ensure_ascii=False)[:200]}...")
    else:
        print("\n✅ 未发现重复样本，所有样本都是唯一的！")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    filepath = '/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_10k_no_clues.jsonl'
    check_duplicates(filepath)
