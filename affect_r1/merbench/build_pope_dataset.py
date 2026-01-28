#!/usr/bin/env python3
"""
POPE压力测试数据集构建脚本

功能：
从GT线索文件中构建POPE问题对，用于测试模型的抗幻觉能力。

两种测试类型：
1. Mask Audio (静音) - 问模型是否听到某个音频线索 → 正确答案是No
2. Mask Visual (黑屏) - 问模型是否看到某个视觉线索 → 正确答案是No

输出格式 (JSONL):
{
    "name": "sample_00000000",
    "video_path": "...",
    "mask_type": "audio",  # "audio" 或 "visual"
    "question": "Does the audio contain the sound of expressing dissatisfaction?",
    "gt_cue": "expressing dissatisfaction and impatience",
    "expected_answer": "no"
}

使用示例：
python -m affect_r1.merbench.build_pope_dataset \
    --gt-clues /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/track3_train_ovmerd_clues_Qwen.jsonl \
    --video-root /mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video \
    --output-dir /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_audio_pope_questions(
    gt_data: List[Dict],
    video_root: str,
    max_cues_per_sample: int = 2
) -> List[Dict]:
    """
    构建音频POPE问题 (Mask Audio场景)
    
    场景：静音音频，问模型是否听到某个音频线索
    正确答案：No（因为音频被静音了）
    """
    questions = []
    
    for item in tqdm(gt_data, desc="构建Audio POPE问题"):
        name = item['name']
        audio_cues = item.get('audio_cues', [])
        
        if not audio_cues:
            continue
        
        # 构建视频路径
        video_path = os.path.join(video_root, f"{name}.mp4")
        
        # 每个样本选择若干个线索构建问题
        selected_cues = audio_cues[:max_cues_per_sample]
        
        for cue in selected_cues:
            if not cue or len(cue.strip()) < 3:
                continue
            
            # 构建问题
            question = f"Does the audio contain {cue}? Please answer with only 'yes' or 'no'."
            
            questions.append({
                "name": name,
                "video_path": video_path,
                "mask_type": "audio",
                "question": question,
                "gt_cue": cue,
                "expected_answer": "no"
            })
    
    return questions


def build_visual_pope_questions(
    gt_data: List[Dict],
    video_root: str,
    max_cues_per_sample: int = 2
) -> List[Dict]:
    """
    构建视觉POPE问题 (Mask Visual场景)
    
    场景：黑屏视频，问模型是否看到某个视觉线索
    正确答案：No（因为视频被黑屏了）
    """
    questions = []
    
    for item in tqdm(gt_data, desc="构建Visual POPE问题"):
        name = item['name']
        visual_cues = item.get('visual_cues', [])
        
        if not visual_cues:
            continue
        
        # 构建视频路径
        video_path = os.path.join(video_root, f"{name}.mp4")
        
        # 每个样本选择若干个线索构建问题
        selected_cues = visual_cues[:max_cues_per_sample]
        
        for cue in selected_cues:
            if not cue or len(cue.strip()) < 3:
                continue
            
            # 构建问题
            question = f"Does the video show {cue}? Please answer with only 'yes' or 'no'."
            
            questions.append({
                "name": name,
                "video_path": video_path,
                "mask_type": "visual",
                "question": question,
                "gt_cue": cue,
                "expected_answer": "no"
            })
    
    return questions


def main():
    parser = argparse.ArgumentParser(description="POPE压力测试数据集构建")
    parser.add_argument("--gt-clues", type=str, required=True,
                        help="GT线索文件路径")
    parser.add_argument("--video-root", type=str, required=True,
                        help="视频文件根目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--max-cues-per-sample", type=int, default=2,
                        help="每个样本最多使用的线索数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("POPE压力测试数据集构建")
    print("=" * 60)
    print(f"GT线索文件: {args.gt_clues}")
    print(f"视频根目录: {args.video_root}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 加载GT线索
    print("[1/3] 加载GT线索...")
    gt_data = load_jsonl(args.gt_clues)
    print(f"  加载样本数: {len(gt_data)}")
    
    # 构建Audio POPE问题
    print("\n[2/3] 构建Audio POPE问题 (Mask Audio场景)...")
    audio_questions = build_audio_pope_questions(
        gt_data, args.video_root, args.max_cues_per_sample
    )
    print(f"  生成问题数: {len(audio_questions)}")
    
    # 构建Visual POPE问题
    print("\n[3/3] 构建Visual POPE问题 (Mask Visual场景)...")
    visual_questions = build_visual_pope_questions(
        gt_data, args.video_root, args.max_cues_per_sample
    )
    print(f"  生成问题数: {len(visual_questions)}")
    
    # 保存数据集
    audio_output = output_dir / "pope_audio.jsonl"
    with open(audio_output, 'w', encoding='utf-8') as f:
        for q in audio_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f"\nAudio POPE数据集已保存: {audio_output}")
    
    visual_output = output_dir / "pope_visual.jsonl"
    with open(visual_output, 'w', encoding='utf-8') as f:
        for q in visual_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f"Visual POPE数据集已保存: {visual_output}")
    
    # 保存统计信息
    stats = {
        "gt_samples": len(gt_data),
        "audio_questions": len(audio_questions),
        "visual_questions": len(visual_questions),
        "max_cues_per_sample": args.max_cues_per_sample,
    }
    
    stats_file = output_dir / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存: {stats_file}")
    
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

