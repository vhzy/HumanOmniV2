#!/usr/bin/env python3
"""
根据track3_train_ovmerd_clues_Qwen.jsonl中的视觉线索生成类似pope_visual.jsonl格式的问题
"""
import json
from pathlib import Path

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

def generate_visual_questions(input_file, output_file, video_base_path="/mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video"):
    """
    从包含视觉线索的文件生成POPE格式的问题
    
    Args:
        input_file: 输入文件路径（track3_train_ovmerd_clues_Qwen.jsonl）
        output_file: 输出文件路径
        video_base_path: 视频文件的基础路径
    """
    print("=" * 80)
    print("【读取输入文件】")
    print("=" * 80)
    
    input_data = load_jsonl(input_file)
    print(f"输入文件: {input_file}")
    print(f"样本数: {len(input_data)}")
    
    # 统计视觉线索总数
    total_visual_cues = sum(len(item.get('visual_cues', [])) for item in input_data)
    print(f"视觉线索总数: {total_visual_cues}\n")
    
    print("=" * 80)
    print("【生成问题】")
    print("=" * 80)
    
    output_data = []
    
    for item in input_data:
        name = item['name']
        visual_cues = item.get('visual_cues', [])
        
        # 构建视频路径
        video_path = f"{video_base_path}/{name}.mp4"
        
        # 为每个视觉线索生成一个问题
        for cue in visual_cues:
            question_item = {
                "name": name,
                "video_path": video_path,
                "mask_type": "visual",
                "question": f"Does the video show {cue}? Please answer with only 'yes' or 'no'.",
                "gt_cue": cue,
                "expected_answer": "yes"  # 因为这些是真实的线索，所以答案是yes
            }
            output_data.append(question_item)
    
    print(f"生成的问题数: {len(output_data)}")
    
    # 保存到输出文件
    save_jsonl(output_data, output_file)
    print(f"\n✓ 已保存到: {output_file}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("【统计信息】")
    print("=" * 80)
    
    # 统计每个样本的问题数量
    sample_counts = {}
    for item in output_data:
        name = item['name']
        sample_counts[name] = sample_counts.get(name, 0) + 1
    
    print(f"样本数: {len(sample_counts)}")
    print(f"总问题数: {len(output_data)}")
    print(f"平均每个样本的问题数: {len(output_data) / len(sample_counts):.2f}")
    
    # 显示前几个样本的统计
    print("\n前10个样本的问题数量:")
    for i, (name, count) in enumerate(list(sample_counts.items())[:10], 1):
        print(f"  {i:2d}. {name}: {count} 个问题")
    
    # 显示前几个生成的问题作为示例
    print("\n" + "=" * 80)
    print("【示例问题】")
    print("=" * 80)
    
    for i, item in enumerate(output_data[:5], 1):
        print(f"\n示例 {i}:")
        print(f"  样本名称: {item['name']}")
        print(f"  视频路径: {item['video_path']}")
        print(f"  问题类型: {item['mask_type']}")
        print(f"  问题: {item['question']}")
        print(f"  线索: {item['gt_cue']}")
        print(f"  期望答案: {item['expected_answer']}")
    
    print("\n" + "=" * 80)
    return output_data


if __name__ == "__main__":
    # 输入文件
    input_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl"
    
    # 输出文件
    output_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_visual_from_train.jsonl"
    
    # 生成问题
    results = generate_visual_questions(input_file, output_file)
    
    print(f"\n✓ 完成！共生成 {len(results)} 个问题")

