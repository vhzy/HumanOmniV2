#!/usr/bin/env python3
"""
POPE压力测试推理脚本 - Early Answer版本

功能：
1. 加载POPE问题对数据集
2. 加载模型并进行推理
3. Mask对应的模态（音频或视觉）
4. 使用early answer方法，预填充空的<context></context><think></think><answer>，让模型直接回答
5. 保存模型回答

使用示例：
CUDA_VISIBLE_DEVICES=1 python -m affect_r1.merbench.run_pope_inference_early_answer \
    --model-path /mnt/afs/hanzhiyuan/huggingface/humanomniv2 \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered.jsonl \
    --output /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_early_a.jsonl \
    --mask-type audio

CUDA_VISIBLE_DEVICES=1 python -m affect_r1.merbench.run_pope_inference_early_answer \
    --model-path /mnt/afs/hanzhiyuan/huggingface/humanomniv2 \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_visual_filtered.jsonl \
    --output /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_early_v.jsonl \
    --mask-type visual

CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.run_pope_inference_early_answer \
    --model-path /mnt/afs/hanzhiyuan/huggingface/humanomniv2 \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered_v2.jsonl\
    --output /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_a4.jsonl \
    --mask-type audio



CUDA_VISIBLE_DEVICES=0 python -m affect_r1.merbench.run_pope_inference_early_answer \
    --model-path /mnt/afs/hanzhiyuan/huggingface/humanomniv2 \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_visual_filtered_v2.jsonl\
    --output  /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_all_v4.jsonl \
    --mask-type visual
"""

import argparse
import json
import os
import re
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal")
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2")

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 从 papo_utils 导入 mask 函数
from affect_r1.papo_utils import mask_visual_inputs, mask_audio_inputs


# ==================== 配置 ====================
POPE_SYSTEM_PROMPT = """You are a helpful assistant analyzing multimedia content. 
Answer the question based ONLY on what you can actually perceive in video or audio.
If the modality is missing, not clear or different from the question, you should answer 'no'.
If the content of the question is included in the video or audio, you should answer 'yes'.
Answer with only 'yes' or 'no'.
"""

# Early Answer 预填充的前缀 - 空的context和think，模型直接在answer后输出
EARLY_ANSWER_PREFIX = "<context> </context><think> </think><answer>"


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
        attn_impl = "sdpa"  # Use SDPA as fallback
    
    # 使用 Thinker 模型（不加载 Talker/Token2Wav）
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    ).eval()
    
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    
    return model, processor


def apply_mask(multimodal_inputs: Dict[str, Any], mask_type: str, mask_ratio: float = 0.9) -> Dict[str, Any]:
    """应用模态mask"""
    if mask_type == "audio":
        return mask_audio_inputs(multimodal_inputs, mask_ratio=mask_ratio, noise=False)
    elif mask_type == "visual":
        return mask_visual_inputs(multimodal_inputs, mask_ratio=mask_ratio, noise=False)
    else:
        return multimodal_inputs


def get_audio_path_from_video(video_path: str) -> str:
    """从视频路径推导音频路径"""
    # video: /mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video/sample_00000000.mp4
    # audio: /mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/audio/sample_00000000.wav
    video_dir = os.path.dirname(video_path)
    audio_dir = video_dir.replace("/video", "/audio")
    name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(audio_dir, f"{name}.wav")


def build_messages(video_path: str, audio_path: str, question: str, early_answer: bool = True) -> List[Dict]:
    """
    构建消息格式（使用单独的音频文件）
    
    Args:
        video_path: 视频路径
        audio_path: 音频路径
        question: 问题文本
        early_answer: 是否使用early answer模式
        
    Returns:
        消息列表
    """
    # 使用单独的音频文件，use_audio_in_video=False
    content = [
        {"type": "video", "video": video_path},
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": f"{question}"}
    ]
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": POPE_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    
    # Early answer mode: 添加 assistant 消息，预填充空的 context/think 标签
    # 这会被放在 <|im_start|>assistant\n 之后
    if early_answer:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": EARLY_ANSWER_PREFIX}]
        })
    
    return messages


def prepare_inputs(processor, messages: List[Dict], use_audio_in_video: bool = False):
    """
    准备模型输入，支持early answer模式
    
    Args:
        processor: 处理器
        messages: 消息列表
        use_audio_in_video: 是否使用视频中的音频
        
    Returns:
        inputs: 模型输入
        text: 处理后的文本
    """
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    
    # 检查是否有 assistant 消息（early answer 模式）
    has_assistant_message = messages[-1]["role"] == "assistant"
    add_gen_prompt = not has_assistant_message
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen_prompt)
    
    # Ensure text is a string
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    
    # Early answer 模式：移除末尾的 <|im_end|> 标签
    # 我们希望模型从预填充的 <answer> 标签后继续生成
    if has_assistant_message:
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
    
    inputs = processor(
        text=[text],
        images=images,
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return inputs, text


def extract_answer(generated_text: str) -> str:
    """
    从生成的文本中提取yes/no答案
    
    对于 early answer 模式，生成的文本应该是直接的答案（因为 <answer> 标签已预填充）
    """
    text_lower = generated_text.lower().strip()
    
    # 移除可能的 </answer> 结束标签
    text_lower = text_lower.replace("</answer>", "").strip()
    
    # 直接匹配
    if text_lower in ["yes", "no", "yes.", "no."]:
        return text_lower.rstrip(".")
    
    # 查找 yes 或 no
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    
    # 如果都有或都没有，返回原始文本前50个字符
    return text_lower[:50]


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="POPE压力测试推理 - Early Answer版本")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--processor-path", type=str, default=None,
                        help="处理器路径（默认与模型路径相同）")
    parser.add_argument("--pope-dataset", type=str, required=True,
                        help="POPE问题对数据集路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--mask-type", type=str, choices=["audio", "visual"], required=True,
                        help="mask类型：audio 或 visual")
    parser.add_argument("--mask-ratio", type=float, default=1.0,
                        help="mask比例（默认1.0）")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="最大生成token数（early answer模式下应该很短）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大处理样本数（用于调试）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑")
    parser.add_argument("--no-early-answer", action="store_true",
                        help="禁用early answer模式（使用原始方式）")
    
    args = parser.parse_args()
    
    use_early_answer = not args.no_early_answer
    
    print("=" * 60)
    print("POPE压力测试推理 - Early Answer版本")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.pope_dataset}")
    print(f"Mask类型: {args.mask_type}")
    print(f"Mask比例: {args.mask_ratio}")
    print(f"Early Answer模式: {'启用' if use_early_answer else '禁用'}")
    print(f"输出: {args.output}")
    print()
    
    if use_early_answer:
        print(f"[INFO] Early Answer预填充: {EARLY_ANSWER_PREFIX}")
    
    # 加载模型
    print("[1/3] 加载模型...")
    model, processor = load_model(args.model_path, args.processor_path)
    
    # 加载数据集
    print("\n[2/3] 加载POPE数据集...")
    pope_data = load_jsonl(args.pope_dataset)
    if args.max_samples:
        pope_data = pope_data[:args.max_samples]
    print(f"  问题数: {len(pope_data)}")
    
    # 断点续跑
    processed_keys = set()
    if args.resume and os.path.exists(args.output):
        print(f"[INFO] 从断点续跑: {args.output}")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = f"{obj['name']}_{obj['gt_cue']}"
                    processed_keys.add(key)
                except:
                    pass
        print(f"[INFO] 已处理: {len(processed_keys)} 个问题")
    
    # 过滤待处理的问题
    questions_to_process = []
    for q in pope_data:
        key = f"{q['name']}_{q['gt_cue']}"
        if args.resume and key in processed_keys:
            continue
        questions_to_process.append(q)
    
    print(f"  待处理问题数: {len(questions_to_process)}")
    
    # 推理统计
    total_count = 0
    correct_count = 0
    
    # 推理
    print("\n[3/3] 开始推理...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for item in tqdm(questions_to_process, desc="推理中"):
            name = item['name']
            video_path = item['video_path']
            question = item['question']
            gt_cue = item['gt_cue']
            expected_answer = item['expected_answer']
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"[WARNING] 视频不存在: {video_path}")
                continue
            
            # 获取音频路径
            audio_path = get_audio_path_from_video(video_path)
            if not os.path.exists(audio_path):
                print(f"[WARNING] 音频不存在: {audio_path}，跳过")
                continue
            
            try:
                # 构建消息（使用单独的音频文件，启用early answer）
                messages = build_messages(video_path, audio_path, question, early_answer=use_early_answer)
                
                # 准备输入
                inputs, text = prepare_inputs(processor, messages, use_audio_in_video=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # 应用mask
                # inputs = apply_mask(inputs, args.mask_type, args.mask_ratio)
                
                # 生成回答
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,  # 使用贪婪解码
                        # do_sample=True,
                        # temperature=0.9,
                        # top_p=0.9,
                        use_audio_in_video=False,  # 使用单独的音频文件
                    )
                
                # 解码输出
                input_len = inputs['input_ids'].shape[1]
                generated_ids = generated_ids[:, input_len:]
                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # 提取答案
                answer = extract_answer(generated_text)
                is_correct = (answer == expected_answer)
                
                total_count += 1
                if is_correct:
                    correct_count += 1
                
                result = {
                    "name": name,
                    "mask_type": args.mask_type,
                    "question": question,
                    "gt_cue": gt_cue,
                    "expected_answer": expected_answer,
                    "model_answer": answer,
                    "full_response": generated_text,
                    "is_correct": is_correct,
                    "early_answer_mode": use_early_answer,
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"[ERROR] 处理 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 打印统计
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"\n推理完成！")
    print(f"总计: {total_count}, 正确: {correct_count}, 准确率: {accuracy:.2f}%")
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

