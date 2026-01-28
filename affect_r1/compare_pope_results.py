#!/usr/bin/env python3
"""
对比两个POPE结果文件，找出不同类型的样本
"""
import json
from collections import defaultdict
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> Dict[str, dict]:
    """
    加载JSONL文件并以(name, question)为key构建字典
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 使用name和question的组合作为唯一标识
            key = (item['name'], item['question'])
            data[key] = item
    return data

def compare_results(file1: str, file2: str):
    """
    对比两个POPE结果文件
    """
    print(f"加载文件1: {file1}")
    data1 = load_jsonl(file1)
    print(f"文件1包含 {len(data1)} 个样本\n")
    
    print(f"加载文件2: {file2}")
    data2 = load_jsonl(file2)
    print(f"文件2包含 {len(data2)} 个样本\n")
    
    # 找到共同的key
    common_keys = set(data1.keys()) & set(data2.keys())
    print(f"共同样本数: {len(common_keys)}\n")
    
    # 分类样本
    both_wrong = []  # 两者都错
    file1_right_file2_wrong = []  # 前者对，后者错
    file1_wrong_file2_right = []  # 前者错，后者对
    both_right = []  # 两者都对
    
    for key in common_keys:
        item1 = data1[key]
        item2 = data2[key]
        
        is_correct1 = item1['is_correct']
        is_correct2 = item2['is_correct']
        
        sample_info = {
            'name': item1['name'],
            'mask_type': item1['mask_type'],
            'question': item1['question'],
            'gt_cue': item1['gt_cue'],
            'expected_answer': item1['expected_answer'],
            'file1_answer': item1['model_answer'],
            'file1_response': item1['full_response'],
            'file1_correct': is_correct1,
            'file2_answer': item2['model_answer'],
            'file2_response': item2['full_response'],
            'file2_correct': is_correct2,
        }
        
        if not is_correct1 and not is_correct2:
            both_wrong.append(sample_info)
        elif is_correct1 and not is_correct2:
            file1_right_file2_wrong.append(sample_info)
        elif not is_correct1 and is_correct2:
            file1_wrong_file2_right.append(sample_info)
        else:  # both correct
            both_right.append(sample_info)
    
    # 打印统计信息
    print("=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"1. 两者都错了的样本数: {len(both_wrong)}")
    print(f"2. 前者对，后者错的样本数: {len(file1_right_file2_wrong)}")
    print(f"3. 后者对，前者错的样本数: {len(file1_wrong_file2_right)}")
    print(f"4. 两者都对的样本数: {len(both_right)}")
    print(f"\n总计: {len(both_wrong) + len(file1_right_file2_wrong) + len(file1_wrong_file2_right) + len(both_right)}")
    
    # 计算准确率
    total = len(common_keys)
    acc1 = (len(file1_right_file2_wrong) + len(both_right)) / total * 100
    acc2 = (len(file1_wrong_file2_right) + len(both_right)) / total * 100
    print(f"\n文件1准确率: {acc1:.2f}%")
    print(f"文件2准确率: {acc2:.2f}%")
    print("=" * 80)
    
    # 保存详细结果到文件
    output_dir = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1"
    
    # 保存类型1：两者都错
    output_file = f"{output_dir}/comparison_both_wrong.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in both_wrong:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n已保存：两者都错的样本 -> {output_file}")
    
    # 保存类型2：前者对，后者错
    output_file = f"{output_dir}/comparison_file1_right_file2_wrong.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in file1_right_file2_wrong:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存：前者对后者错的样本 -> {output_file}")
    
    # 保存类型3：后者对，前者错
    output_file = f"{output_dir}/comparison_file1_wrong_file2_right.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in file1_wrong_file2_right:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存：后者对前者错的样本 -> {output_file}")
    
    # 保存类型4：两者都对（可选）
    output_file = f"{output_dir}/comparison_both_right.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in both_right:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存：两者都对的样本 -> {output_file}")
    
    # 打印一些示例
    print("\n" + "=" * 80)
    print("示例预览:")
    print("=" * 80)
    
    if both_wrong:
        print("\n【示例1：两者都错】")
        sample = both_wrong[0]
        print(f"样本名称: {sample['name']}")
        print(f"问题: {sample['question']}")
        print(f"期望答案: {sample['expected_answer']}")
        print(f"文件1答案: {sample['file1_answer']} (错误)")
        print(f"文件2答案: {sample['file2_answer']} (错误)")
    
    if file1_right_file2_wrong:
        print("\n【示例2：前者对，后者错】")
        sample = file1_right_file2_wrong[0]
        print(f"样本名称: {sample['name']}")
        print(f"问题: {sample['question']}")
        print(f"期望答案: {sample['expected_answer']}")
        print(f"文件1答案: {sample['file1_answer']} (正确)")
        print(f"文件2答案: {sample['file2_answer']} (错误)")
    
    if file1_wrong_file2_right:
        print("\n【示例3：后者对，前者错】")
        sample = file1_wrong_file2_right[0]
        print(f"样本名称: {sample['name']}")
        print(f"问题: {sample['question']}")
        print(f"期望答案: {sample['expected_answer']}")
        print(f"文件1答案: {sample['file1_answer']} (错误)")
        print(f"文件2答案: {sample['file2_answer']} (正确)")
    
    print("\n" + "=" * 80)
    
    return {
        'both_wrong': both_wrong,
        'file1_right_file2_wrong': file1_right_file2_wrong,
        'file1_wrong_file2_right': file1_wrong_file2_right,
        'both_right': both_right
    }


if __name__ == "__main__":
    file1 = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_stage2_2/pope_results_a2.jsonl"
    file2 = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo23_v2/pope_results_a2.jsonl"
    
    results = compare_results(file1, file2)

