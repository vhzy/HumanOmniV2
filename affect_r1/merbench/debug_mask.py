#!/usr/bin/env python3
"""
调试脚本：检查 Qwen2.5-Omni 的输入 keys 和 mask 是否生效

CUDA_VISIBLE_DEVICES=0 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/debug_mask.py
"""

import sys
import torch
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2/src/open-r1-multimodal")
sys.path.insert(0, "/mnt/afs/hanzhiyuan/code/HumanOmniV2")

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from affect_r1.papo_utils import mask_audio_inputs, AUDIO_MASK_KEYWORDS, SKIP_KEYWORDS
import importlib.util

# 测试样本
VIDEO_PATH = "/mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/video/sample_00000000.mp4"
AUDIO_PATH = "/mnt/afs/hanzhiyuan/datasets/mer2025/ovmerdplus-process/audio/sample_00000000.wav"


def run_inference(model, processor, inputs, question_desc):
    """运行推理并打印结果"""
    inputs_device = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs_device,
            max_new_tokens=20,
            do_sample=False,
            use_audio_in_video=False,
        )
    
    input_len = inputs_device['input_ids'].shape[1]
    generated_ids = generated_ids[:, input_len:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"  [{question_desc}] 模型回答: {response.strip()}")
    return response


def main():
    print("=" * 60)
    print("调试 Qwen2.5-Omni 音频 Mask - 完整测试")
    print("=" * 60)
    
    # 检查 flash attention
    try:
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    
    # 加载模型和 processor
    print("\n[1] 加载模型和 processor...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B")
    
    # ==================== 测试 1：正常输入，问能否听到音频 ====================
    print("\n" + "=" * 60)
    print("[测试 1] 正常输入（有音频），问：Can you hear audio?")
    print("=" * 60)
    
    content = [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "audio", "audio": AUDIO_PATH},
        {"type": "text", "text": "Can you hear any audio or speech? Answer with only 'yes' or 'no'."}
    ]
    # 使用与 run_pope_inference_v2.py 相同的 POPE_SYSTEM_PROMPT
    POPE_SYSTEM_PROMPT = """
You are a helpful assistant analyzing multimedia content. Answer the question based ONLY on what you can actually perceive in video or audio.
If the modality is missing, not clear or different from the question, you should answer 'no'. 
If you believe that the input multimodal content simply needs to contain the modality described in the problem (e.g., frowning corresponds to the visual modality), you do not need to consider specific details; please prioritize outputting 'yes'. 
Answer with only 'yes' or 'no'."""
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": POPE_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]
    
    inputs = processor(text=[text], images=images, audio=audios, videos=videos, 
                      return_tensors="pt", padding=True, use_audio_in_video=False)
    
    run_inference(model, processor, dict(inputs), "正常音频")
    
    # ==================== 测试 2：Mask 音频后，问能否听到 ====================
    print("\n" + "=" * 60)
    print("[测试 2] Mask 音频后（input_features=0），问：Can you hear audio?")
    print("=" * 60)
    
    inputs_masked = mask_audio_inputs(dict(inputs), mask_ratio=1.0, noise=False)
    
    # 验证 mask 是否生效
    print(f"  input_features 原始 mean: {inputs['input_features'].mean().item():.6f}")
    print(f"  input_features mask后 mean: {inputs_masked['input_features'].mean().item():.6f}")
    
    run_inference(model, processor, inputs_masked, "Mask 音频")
    
    # ==================== 测试 3：完全不传音频，问能否听到 ====================
    print("\n" + "=" * 60)
    print("[测试 3] 完全不传音频文件，问：Can you hear audio?")
    print("=" * 60)
    
    content_no_audio = [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "text", "text": "Can you hear any audio or speech? Answer with only 'yes' or 'no'."}
    ]
    messages_no_audio = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content_no_audio}
    ]
    
    audios2, images2, videos2 = process_mm_info(messages_no_audio, use_audio_in_video=False)
    text2 = processor.apply_chat_template(messages_no_audio, tokenize=False, add_generation_prompt=True)
    if isinstance(text2, list):
        text2 = text2[0]
    
    inputs_no_audio = processor(text=[text2], images=images2, audio=audios2, videos=videos2, 
                                return_tensors="pt", padding=True, use_audio_in_video=False)
    
    print(f"  input_features 存在: {'input_features' in inputs_no_audio}")
    if 'input_features' in inputs_no_audio:
        print(f"  input_features shape: {inputs_no_audio['input_features'].shape}")
    
    run_inference(model, processor, dict(inputs_no_audio), "无音频文件")
    
    # ==================== 测试 4：Mask 音频后，问能否看到视频 ====================
    print("\n" + "=" * 60)
    print("[测试 4] Mask 音频后，问：Can you see the video? (应该回答 yes)")
    print("=" * 60)
    
    content_video = [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "audio", "audio": AUDIO_PATH},
        {"type": "text", "text": "Can you see the video? Answer with only 'yes' or 'no'."}
    ]
    messages_video = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content_video}
    ]
    
    audios3, images3, videos3 = process_mm_info(messages_video, use_audio_in_video=False)
    text3 = processor.apply_chat_template(messages_video, tokenize=False, add_generation_prompt=True)
    if isinstance(text3, list):
        text3 = text3[0]
    
    inputs_video = processor(text=[text3], images=images3, audio=audios3, videos=videos3, 
                            return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs_video_masked = mask_audio_inputs(dict(inputs_video), mask_ratio=1.0, noise=False)
    
    run_inference(model, processor, inputs_video_masked, "Mask音频，问视频")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

