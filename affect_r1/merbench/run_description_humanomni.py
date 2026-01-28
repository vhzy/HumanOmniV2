#!/usr/bin/env python3
"""
HumanOmniV2 情感描述生成脚本

功能：
1. 加载 HumanOmniV2 模型
2. 对 OVMERDPlus 数据集生成情感描述
3. 输出 JSONL 格式，供 check_hallucination.py 提取线索

使用示例：
CUDA_VISIBLE_DEVICES=0 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/run_description_humanomni.py \
    --model-path /mnt/afs/hanzhiyuan/huggingface/humanomniv2 \
    --output /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/humanomniv2_baseline/ovmerd_description.jsonl \
    --dataset-root /mnt/afs/hanzhiyuan/datasets

或使用已训练的 checkpoint：
CUDA_VISIBLE_DEVICES=0 python run_description_humanomni.py \
    --model-path /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo_beta10_v2 \
    --processor-path /mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B \
    --output /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo_beta10_v2/ovmerd_description.jsonl \
    --dataset-root /mnt/afs/hanzhiyuan/datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal")
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2")

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import av


# ==================== 配置 ====================
AFFECT_SYSTEM_PROMPT = (
    "You are an expert affective-computing assistant. "
    "Please infer the person's emotional state and provide your reasoning process."
)


def check_if_video_has_audio(video_path: str) -> bool:
    """检查视频是否包含音频流"""
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        return len(audio_streams) > 0
    except:
        return False


def load_model(model_path: str, processor_path: str = None):
    """加载模型和处理器"""
    if processor_path is None:
        processor_path = model_path
    
    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] Loading processor from: {processor_path}")
    
    # 检查 flash attention 是否可用
    import importlib.util
    try:
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    ).eval()
    
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    
    return model, processor


def load_ovmerdplus_data(dataset_root: str) -> tuple:
    """加载 OVMERDPlus 数据集"""
    ovmerd_root = os.path.join(dataset_root, "mer2025", "ovmerdplus-process")
    
    # 加载字幕
    subtitle_csv = os.path.join(ovmerd_root, "subtitle_eng.csv")
    name2subtitle = {}
    if os.path.exists(subtitle_csv):
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = str(row['name']).strip()
            subtitle = row.get('sentence', row.get('english', ''))
            if pd.isna(subtitle):
                subtitle = ""
            name2subtitle[name] = str(subtitle)
    
    # 获取测试样本名称
    test_names = list(name2subtitle.keys())
    
    video_dir = os.path.join(ovmerd_root, "video")
    audio_dir = os.path.join(ovmerd_root, "audio")
    
    return test_names, name2subtitle, video_dir, audio_dir


def build_prompt_text(subtitle: str) -> str:
    """构建 prompt 文本"""
    subtitle_prompt = ""
    if subtitle and subtitle.strip():
        subtitle_prompt = f"\nThe subtitle of this video is: <Subtitle>{subtitle.strip()}</Subtitle>."
    
    question = (
        "As an expert in the field of emotions, please focus on the facial expressions, body movements, tone, "
        "subtitle content, etc., in the video to discern clues related to the emotions of the individual. "
        "Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
    )
    
    return f"{subtitle_prompt}\n{question}"


def build_messages(video_path: str, audio_path: str, subtitle: str, use_audio_in_video: bool = True) -> List[Dict]:
    """构建消息格式"""
    text_prompt = build_prompt_text(subtitle)
    
    has_separate_audio = audio_path is not None and os.path.exists(audio_path)
    
    if has_separate_audio:
        content = [
            {"type": "video", "video": video_path},
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"Here is a video, with the audio.\n{text_prompt}"}
        ]
    elif use_audio_in_video and video_path and check_if_video_has_audio(video_path):
        content = [
            {"type": "video", "video": video_path},
            {"type": "audio", "audio": video_path},
            {"type": "text", "text": f"Here is a video, with the audio from the video.\n{text_prompt}"}
        ]
    else:
        content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": f"Here is the video.\n{text_prompt}"}
        ]
    
    return [
        {"role": "system", "content": [{"type": "text", "text": AFFECT_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def prepare_inputs(processor, messages: List[Dict], use_audio_in_video: bool = False):
    """准备模型输入"""
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    
    inputs = processor(
        text=[text],
        images=images,
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return inputs


def main():
    parser = argparse.ArgumentParser(description="HumanOmniV2 情感描述生成")
    parser.add_argument("--model-path", type=str, required=True,
                        help="HumanOmniV2 模型路径")
    parser.add_argument("--processor-path", type=str, default=None,
                        help="处理器路径（默认与模型路径相同）")
    parser.add_argument("--dataset-root", type=str, default="/mnt/afs/hanzhiyuan/datasets",
                        help="数据集根目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p 采样")
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="是否使用采样")
    parser.add_argument("--use-audio-in-video", action="store_true", default=True,
                        help="使用视频中的音频")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大处理样本数（用于调试）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HumanOmniV2 情感描述生成")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"数据集根目录: {args.dataset_root}")
    print(f"输出: {args.output}")
    print()
    
    # 加载模型
    print("[1/3] 加载模型...")
    model, processor = load_model(args.model_path, args.processor_path)
    
    # 加载数据集
    print("\n[2/3] 加载 OVMERDPlus 数据集...")
    test_names, name2subtitle, video_dir, audio_dir = load_ovmerdplus_data(args.dataset_root)
    
    if args.max_samples:
        test_names = test_names[:args.max_samples]
    print(f"  样本数: {len(test_names)}")
    
    # 断点续跑
    processed_names = set()
    if args.resume and os.path.exists(args.output):
        print(f"[INFO] 从断点续跑: {args.output}")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    name = obj.get("name")
                    if name:
                        processed_names.add(name)
                except:
                    pass
        print(f"[INFO] 已处理: {len(processed_names)} 个样本")
    
    # 过滤待处理的样本
    names_to_process = [n for n in test_names if n not in processed_names]
    print(f"  待处理样本数: {len(names_to_process)}")
    
    # 推理
    print("\n[3/3] 开始生成描述...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for name in tqdm(names_to_process, desc="生成中"):
            video_path = os.path.join(video_dir, f"{name}.mp4")
            audio_path = os.path.join(audio_dir, f"{name}.wav")
            subtitle = name2subtitle.get(name, "")
            
            if not os.path.exists(video_path):
                print(f"[WARNING] 视频不存在: {video_path}")
                continue
            
            # 判断是否有单独的音频文件
            has_separate_audio = os.path.exists(audio_path)
            use_audio_in_video_for_processing = False if has_separate_audio else args.use_audio_in_video
            
            try:
                messages = build_messages(
                    video_path,
                    audio_path if has_separate_audio else None,
                    subtitle,
                    use_audio_in_video=args.use_audio_in_video
                )
                
                inputs = prepare_inputs(processor, messages, use_audio_in_video_for_processing)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_audio_in_video=use_audio_in_video_for_processing,
                    )
                
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[0]
                response_ids = generated[input_len:] if generated.shape[0] > input_len else generated
                response = processor.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                # 输出 JSONL 格式（与 check_hallucination.py 兼容）
                result = {
                    "name": name,
                    "think": response,  # 直接使用 response 作为 think 内容
                    "metadata": {
                        "counterfactual_config": {
                            "mask_modality": "none"  # baseline 模式
                        }
                    }
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"[ERROR] 处理 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n生成完成！结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

