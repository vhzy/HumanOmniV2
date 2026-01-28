#!/usr/bin/env python3
"""
过滤掉 visual_cues 或 audio_cues 为空的样本
"""
import json
import argparse
from pathlib import Path


def filter_empty_cues(input_file, output_file=None, backup=True):
    """
    过滤掉 visual_cues 或 audio_cues 为空的样本
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出文件路径，如果为 None 则覆盖原文件
        backup: 是否备份原文件
    """
    input_path = Path(input_file)
    
    # 如果没有指定输出文件，则覆盖原文件
    if output_file is None:
        output_file = input_file
        if backup:
            backup_file = input_path.with_suffix(input_path.suffix + '.backup')
            print(f"备份原文件到: {backup_file}")
            input_path.rename(backup_file)
            input_file = str(backup_file)
    
    output_path = Path(output_file)
    
    # 统计信息
    total_count = 0
    filtered_count = 0
    empty_visual_count = 0
    empty_audio_count = 0
    empty_both_count = 0
    
    # 读取并过滤数据
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_count += 1
            
            try:
                data = json.loads(line.strip())
                
                visual_cues = data.get('visual_cues', [])
                audio_cues = data.get('audio_cues', [])
                
                # 检查是否为空
                visual_empty = not visual_cues or len(visual_cues) == 0
                audio_empty = not audio_cues or len(audio_cues) == 0
                
                # 如果 visual_cues 或 audio_cues 有任意一个为空，则跳过该样本
                if visual_empty or audio_empty:
                    filtered_count += 1
                    if visual_empty and audio_empty:
                        empty_both_count += 1
                        print(f"过滤样本 (两者都为空): {data.get('name', 'unknown')}")
                    elif visual_empty:
                        empty_visual_count += 1
                        print(f"过滤样本 (visual_cues为空): {data.get('name', 'unknown')}")
                    else:
                        empty_audio_count += 1
                        print(f"过滤样本 (audio_cues为空): {data.get('name', 'unknown')}")
                    continue
                
                # 保留该样本
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"警告: 第 {total_count} 行 JSON 解析错误: {e}")
                continue
    
    # 输出统计信息
    remaining_count = total_count - filtered_count
    print("\n" + "="*60)
    print("过滤完成！")
    print(f"总样本数: {total_count}")
    print(f"过滤样本数: {filtered_count}")
    print(f"  - visual_cues 为空: {empty_visual_count}")
    print(f"  - audio_cues 为空: {empty_audio_count}")
    print(f"  - 两者都为空: {empty_both_count}")
    print(f"保留样本数: {remaining_count}")
    print(f"输出文件: {output_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="过滤掉 visual_cues 或 audio_cues 为空的样本"
    )
    parser.add_argument(
        "input_file",
        help="输入的 JSONL 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（默认覆盖原文件）",
        default=None
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原文件（仅在覆盖原文件时有效）"
    )
    
    args = parser.parse_args()
    
    filter_empty_cues(
        args.input_file,
        args.output,
        backup=not args.no_backup
    )

