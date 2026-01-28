#!/usr/bin/env python3
"""
AffectGPT 版本的 POPE 压力测试推理脚本

功能：
1. 加载 AffectGPT 模型
2. 加载 POPE 问题对数据集
3. Mask 对应的模态（音频或视觉）
4. 保存模型回答

使用示例：
# 方式1：自动查找 checkpoint（根据 ckpt-epoch）
CUDA_VISIBLE_DEVICES=0 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/run_pope_inference_affectgpt.py \
    --cfg-path /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered_v2.jsonl \
    --output /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/pope_results/pope_audio_mask2.jsonl \
    --mask-type audio \
    --ckpt-epoch 30

# 方式2：直接指定 checkpoint 路径
CUDA_VISIBLE_DEVICES=1 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/run_pope_inference_affectgpt.py \
    --cfg-path /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml \
    --ckpt-path /mnt/afs/hanzhiyuan/huggingface/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_audio_filtered_v2.jsonl \
    --output /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/pope_results/pope_audio_mask3.jsonl \
    --mask-type audio

# 测试视觉幻觉：
CUDA_VISIBLE_DEVICES=1 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/run_pope_inference_affectgpt.py \
    --cfg-path /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml \
    --ckpt-path /mnt/afs/hanzhiyuan/huggingface/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth \
    --pope-dataset /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/pope_dataset/pope_visual_filtered_v2.jsonl \
    --output /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/pope_results/pope_visual_mask3.jsonl \
    --mask-type visual 

"""

import os
import sys
import json
import glob
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

# 添加 AffectGPT 路径
AFFECTGPT_ROOT = "/mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT"
AFFECTGPT_CKPT_ROOT = "/mnt/afs/hanzhiyuan/huggingface/AffectGPT"  # checkpoint 存储位置
sys.path.insert(0, AFFECTGPT_ROOT)

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import *

import config as affectgpt_config

# 更新 AffectGPT config 中的数据集路径为绝对路径
DATASET_ROOT = "/mnt/afs/hanzhiyuan/datasets/mer2025"
affectgpt_config.DATA_DIR['OVMERDPlus'] = os.path.join(DATASET_ROOT, 'ovmerdplus-process')
# 同步更新依赖 DATA_DIR 的其他路径
affectgpt_config.PATH_TO_RAW_AUDIO['OVMERDPlus'] = os.path.join(affectgpt_config.DATA_DIR['OVMERDPlus'], 'audio')
affectgpt_config.PATH_TO_RAW_VIDEO['OVMERDPlus'] = os.path.join(affectgpt_config.DATA_DIR['OVMERDPlus'], 'video')
affectgpt_config.PATH_TO_RAW_FACE['OVMERDPlus'] = os.path.join(affectgpt_config.DATA_DIR['OVMERDPlus'], 'openface_face')
affectgpt_config.PATH_TO_TRANSCRIPTIONS['OVMERDPlus'] = os.path.join(affectgpt_config.DATA_DIR['OVMERDPlus'], 'subtitle_eng.csv')
affectgpt_config.PATH_TO_LABEL['OVMERDPlus'] = os.path.join(affectgpt_config.DATA_DIR['OVMERDPlus'], 'ovlabel.csv')


# ==================== POPE System Prompt ====================
# AffectGPT 原始 prompt 格式：user_message 只是问题本身，没有 system prompt
# 所以我们把指令简化，直接拼在问题后面
POPE_INSTRUCTION = "If the modality is missing, you should answer 'no',else you should answer 'yes'. Answer with only 'yes' or 'no'."

# ==================== Sanity Check Instruction ====================
# 用于测试模型是否真的能感知到视频/音频
SANITY_CHECK_INSTRUCTION = "Answer with only 'yes' or 'no'."

SANITY_CHECK_QUESTIONS = {
    "video": "Can you see any visual content in the video? Is there a person visible in the video?",
    "audio": "Can you hear any audio or speech in the provided audio?",
    "both": "Can you see visual content and hear audio in this video?",
}


def get_ckpt_path(ckpt_root: str, epoch: int) -> str:
    """获取指定 epoch 的 checkpoint 路径"""
    pattern = f"{ckpt_root}/*{epoch:06d}*.pth"
    ckpts = glob.glob(pattern)
    if len(ckpts) == 0:
        # 尝试其他格式
        pattern = f"{ckpt_root}/checkpoint_{epoch:06d}*.pth"
        ckpts = glob.glob(pattern)
    if len(ckpts) == 0:
        raise ValueError(f"No checkpoint found for epoch {epoch} in {ckpt_root}")
    if len(ckpts) > 1:
        print(f"[WARNING] Multiple checkpoints found for epoch {epoch}, using: {ckpts[0]}")
    return ckpts[0]


def load_model(cfg_path: str, ckpt_epoch: int, gpu: int = 0, ckpt_path: str = None):
    """加载 AffectGPT 模型
    
    Args:
        cfg_path: 配置文件路径
        ckpt_epoch: checkpoint epoch（如果 ckpt_path 未指定）
        gpu: GPU ID
        ckpt_path: 直接指定 checkpoint 路径（优先级高于 ckpt_epoch）
    """
    # 创建伪参数
    class Args:
        pass
    args = Args()
    args.cfg_path = cfg_path
    args.options = None
    
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    inference_cfg = cfg.inference_cfg
    datasets_cfg = cfg.datasets_cfg
    device = f'cuda:{gpu}'
    
    # 如果直接指定了 ckpt_path，使用它
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Using specified checkpoint: {ckpt_path}")
    else:
        # 查找 checkpoint
        cfg_name = os.path.basename(cfg_path)[:-len('.yaml')]
        
        # 搜索多个可能的路径
        search_paths = [
            os.path.join(AFFECTGPT_CKPT_ROOT, cfg_name, cfg_name+'*'),  # huggingface/AffectGPT/
            os.path.join(AFFECTGPT_ROOT, 'output', cfg_name, cfg_name+'*'),  # AffectGPT/output/
        ]
        
        ckpt_root_candidates = []
        for pattern in search_paths:
            ckpt_root_candidates.extend(glob.glob(pattern))
        
        if len(ckpt_root_candidates) == 0:
            raise ValueError(f"No checkpoint root found for {cfg_name}. Searched:\n  " + "\n  ".join(search_paths))
        
        # 选择文件最多的目录
        ckpt_root = max(ckpt_root_candidates, 
                        key=lambda x: len([p for p in os.listdir(x) if p.startswith('checkpoint_')]))
        print(f"[INFO] Using checkpoint root: {ckpt_root}")
        
        # 获取指定 epoch 的 checkpoint
        ckpt_path = get_ckpt_path(ckpt_root, ckpt_epoch)
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
    
    model_cfg.ckpt_3 = ckpt_path
    
    # 加载模型
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    
    chat = Chat(model, model_cfg, device=device)
    
    # 获取 face_or_frame 类型
    face_or_frame = 'multiface_audio_face_text'  # 默认使用 face
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame = datasets_cfg['mercaptionplus'].face_or_frame
    
    return chat, model_cfg, inference_cfg, face_or_frame, device


def load_dataset_cls(face_or_frame: str, inference_cfg):
    """加载数据集处理类"""
    from my_affectgpt.datasets.datasets.ovmerdplus_dataset import OVMERDPlus_Dataset
    
    dataset_cls = OVMERDPlus_Dataset()
    dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
    dataset_cls.vis_processor = BaseProcessor()
    dataset_cls.img_processor = BaseProcessor()
    
    vis_processor_cfg = inference_cfg.get("vis_processor")
    img_processor_cfg = inference_cfg.get("img_processor")
    if vis_processor_cfg is not None:
        dataset_cls.vis_processor = registry.get_processor_class(
            vis_processor_cfg.train.name
        ).from_config(vis_processor_cfg.train)
    if img_processor_cfg is not None:
        dataset_cls.img_processor = registry.get_processor_class(
            img_processor_cfg.train.name
        ).from_config(img_processor_cfg.train)
    
    return dataset_cls


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_answer(response: str) -> str:
    """从生成的文本中提取 yes/no 答案"""
    text_lower = response.lower().strip()
    
    # 直接匹配
    if text_lower in ["yes", "no"]:
        return text_lower
    
    # 查找 yes 或 no
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    
    # 如果都有或都没有，返回原始文本前50个字符
    return text_lower[:50]


def get_paths_from_pope_item(item: Dict) -> tuple:
    """从 POPE item 中提取路径"""
    video_path = item['video_path']
    # 音频路径：video -> audio
    video_dir = os.path.dirname(video_path)
    audio_dir = video_dir.replace("/video", "/audio")
    name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(audio_dir, f"{name}.wav")
    
    # Face 路径
    face_dir = video_dir.replace("/video", "/openface_face")
    face_path = os.path.join(face_dir, f"{name}.npy")
    
    return video_path, audio_path, face_path


def run_inference(
    chat: Chat,
    dataset_cls,
    item: Dict,
    face_or_frame: str,
    mask_type: str,
    device: str,
    debug: bool = False,
) -> Dict:
    """对单个样本进行推理"""
    name = item['name']
    question = item['question']
    
    video_path, audio_path, face_path = get_paths_from_pope_item(item)
    
    if debug:
        print(f"\n[DEBUG] Processing: {name}")
        print(f"[DEBUG] video_path: {video_path}, exists: {os.path.exists(video_path)}")
        print(f"[DEBUG] audio_path: {audio_path}, exists: {os.path.exists(audio_path)}")
        print(f"[DEBUG] face_path: {face_path}, exists: {os.path.exists(face_path)}")
        print(f"[DEBUG] needed_data: {dataset_cls.needed_data}")
        print(f"[DEBUG] n_frms: {getattr(dataset_cls, 'n_frms', 'NOT SET')}")
    
    # 读取数据
    sample_data = dataset_cls.read_frame_face_audio_text(
        video_path, face_path, audio_path, image_path=None
    )
    
    if debug:
        print(f"[DEBUG] sample_data keys: {sample_data.keys()}")
        for k, v in sample_data.items():
            if v is not None:
                print(f"[DEBUG]   {k}: shape={v.shape if hasattr(v, 'shape') else type(v)}")
            else:
                print(f"[DEBUG]   {k}: None")
    
    # ==================== 在原始特征层面应用 mask ====================
    if mask_type == "audio":
        # Mask 音频：将原始音频数据设为零
        if 'audio' in sample_data and sample_data['audio'] is not None:
            if debug:
                print(f"[DEBUG] Masking audio: shape={sample_data['audio'].shape}")
            sample_data['audio'] = torch.zeros_like(sample_data['audio'])
    elif mask_type == "visual":
        # Mask 视觉：将原始视频和face数据设为零
        if 'video' in sample_data and sample_data['video'] is not None:
            if debug:
                print(f"[DEBUG] Masking video: shape={sample_data['video'].shape}")
            sample_data['video'] = torch.zeros_like(sample_data['video'])
        if 'face' in sample_data and sample_data['face'] is not None:
            if debug:
                print(f"[DEBUG] Masking face: shape={sample_data['face'].shape}")
            sample_data['face'] = torch.zeros_like(sample_data['face'])
    
    # 处理各模态（使用 mask 后的数据）
    audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
    frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
    face_hiddens, face_llms = chat.postprocess_face(sample_data)
    _, image_llms = chat.postprocess_image(sample_data)
    
    if debug:
        print(f"[DEBUG] audio_llms: {audio_llms.shape if audio_llms is not None else None}")
        print(f"[DEBUG] frame_llms: {frame_llms.shape if frame_llms is not None else None}")
        print(f"[DEBUG] face_llms: {face_llms.shape if face_llms is not None else None}")
    
    # Multi fusion（使用 mask 后的 hiddens）
    multi_llms = None
    if face_or_frame.startswith('multiface'):
        _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
    elif face_or_frame.startswith('multiframe'):
        _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)
    
    # 如果有 mask，multi 特征也置零（因为 multi fusion 可能不完全依赖 mask 的模态）
    if mask_type in ["audio", "visual"] and multi_llms is not None:
        if debug:
            print(f"[DEBUG] Zeroing multi_llms due to mask_type={mask_type}")
        multi_llms = torch.zeros_like(multi_llms)
    
    # 构建 img_list
    img_list = {
        'audio': audio_llms,
        'frame': frame_llms,
        'face': face_llms,
        'image': image_llms,
        'multi': multi_llms,
    }
    
    # 构建 prompt
    # 使用 POPE 问题格式
    user_message = f"{question} {POPE_INSTRUCTION}"
    subtitle = ""  # POPE 测试不使用字幕
    prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
    
    if debug:
        print(f"[DEBUG] Prompt: {prompt[:500]}...")
    
    # 推理
    with torch.inference_mode():
        response = chat.answer_sample(
            prompt=prompt,
            img_list=img_list,
            num_beams=1,
            # do_sample=False,  # 使用贪婪解码
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=50,
            max_length=2000,
        )
    
    return response


def run_sanity_check(
    chat: Chat,
    dataset_cls,
    video_path: str,
    audio_path: str,
    face_path: str,
    face_or_frame: str,
    mask_type: str,
    device: str,
    check_type: str = "both",  # "video", "audio", "both"
) -> Dict:
    """
    运行 sanity check，测试模型是否真的能感知到视频/音频
    
    Args:
        check_type: 检查类型 - "video", "audio", "both"
    """
    # 读取数据
    sample_data = dataset_cls.read_frame_face_audio_text(
        video_path, face_path, audio_path, image_path=None
    )
    
    # ==================== 在原始特征层面应用 mask ====================
    if mask_type == "audio":
        # Mask 音频：将原始音频数据设为零
        if 'audio' in sample_data and sample_data['audio'] is not None:
            sample_data['audio'] = torch.zeros_like(sample_data['audio'])
    elif mask_type == "visual":
        # Mask 视觉：将原始视频和face数据设为零
        if 'video' in sample_data and sample_data['video'] is not None:
            sample_data['video'] = torch.zeros_like(sample_data['video'])
        if 'face' in sample_data and sample_data['face'] is not None:
            sample_data['face'] = torch.zeros_like(sample_data['face'])
    
    # 处理各模态（使用 mask 后的数据）
    audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
    frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
    face_hiddens, face_llms = chat.postprocess_face(sample_data)
    _, image_llms = chat.postprocess_image(sample_data)
    
    # Multi fusion（使用 mask 后的 hiddens）
    multi_llms = None
    if face_or_frame.startswith('multiface'):
        _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
    elif face_or_frame.startswith('multiframe'):
        _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)
    
    # 如果有 mask，multi 特征也置零
    if mask_type in ["audio", "visual"] and multi_llms is not None:
        multi_llms = torch.zeros_like(multi_llms)
    
    # 构建 img_list
    img_list = {
        'audio': audio_llms,
        'frame': frame_llms,
        'face': face_llms,
        'image': image_llms,
        'multi': multi_llms,
    }
    
    # 获取 sanity check 问题
    question = SANITY_CHECK_QUESTIONS.get(check_type, SANITY_CHECK_QUESTIONS["both"])
    
    # 构建 prompt
    user_message = f"{question} {SANITY_CHECK_INSTRUCTION}"
    subtitle = ""
    prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
    
    print(f"\n[Sanity Check] Question: {question}")
    print(f"[Sanity Check] Mask type: {mask_type}")
    
    # 推理
    with torch.inference_mode():
        response = chat.answer_sample(
            prompt=prompt,
            img_list=img_list,
            num_beams=1,
            do_sample=False,
            max_new_tokens=100,  # 允许更长的回答
            max_length=2000,
        )
    
    print(f"[Sanity Check] Response: {response}")
    
    return {
        "check_type": check_type,
        "mask_type": mask_type,
        "question": question,
        "response": response,
    }


def main():
    parser = argparse.ArgumentParser(description="AffectGPT POPE 压力测试推理")
    parser.add_argument("--cfg-path", type=str, required=True,
                        help="AffectGPT 配置文件路径")
    parser.add_argument("--ckpt-epoch", type=int, default=30,
                        help="Checkpoint epoch (default: 30)")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="直接指定 checkpoint 路径（优先级高于 ckpt-epoch）")
    parser.add_argument("--pope-dataset", type=str, required=True,
                        help="POPE 问题对数据集路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--mask-type", type=str, choices=["audio", "visual", "none"],
                        required=True, help="mask 类型：audio, visual, none")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (default: 0)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大处理样本数（用于调试）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式，打印详细信息")
    parser.add_argument("--sanity-check", action="store_true",
                        help="运行 sanity check 测试，直接问模型是否能看到/听到内容")
    parser.add_argument("--sanity-check-type", type=str, choices=["video", "audio", "both"],
                        default="both", help="sanity check 类型")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AffectGPT POPE 压力测试推理")
    print("=" * 60)
    print(f"配置文件: {args.cfg_path}")
    print(f"Checkpoint Epoch: {args.ckpt_epoch}")
    if args.ckpt_path:
        print(f"Checkpoint 路径: {args.ckpt_path}")
    print(f"数据集: {args.pope_dataset}")
    print(f"Mask类型: {args.mask_type}")
    print(f"输出: {args.output}")
    print()
    
    # 加载模型
    print("[1/3] 加载 AffectGPT 模型...")
    chat, model_cfg, inference_cfg, face_or_frame, device = load_model(
        args.cfg_path, args.ckpt_epoch, args.gpu, args.ckpt_path
    )
    print(f"  Face/Frame 类型: {face_or_frame}")
    
    # 加载数据集处理类
    print("\n[2/3] 加载数据集处理类...")
    dataset_cls = load_dataset_cls(face_or_frame, inference_cfg)
    dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms
    
    # ==================== Sanity Check 模式 ====================
    if args.sanity_check:
        print("\n" + "=" * 60)
        print("运行 Sanity Check 测试")
        print("=" * 60)
        
        # 加载一个样本进行测试
        pope_data = load_jsonl(args.pope_dataset)
        if len(pope_data) == 0:
            print("[ERROR] 数据集为空")
            return
        
        # 取前几个样本进行测试
        test_samples = pope_data[:min(3, len(pope_data))]
        
        results = []
        for i, item in enumerate(test_samples):
            video_path, audio_path, face_path = get_paths_from_pope_item(item)
            
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                print(f"[WARNING] 文件不存在，跳过: {item['name']}")
                continue
            
            print(f"\n--- 测试样本 {i+1}: {item['name']} ---")
            
            # 测试不同的 mask 类型
            for mask in ["none", "visual", "audio"]:
                result = run_sanity_check(
                    chat, dataset_cls, video_path, audio_path, face_path,
                    face_or_frame, mask, device, args.sanity_check_type
                )
                result["sample_name"] = item['name']
                results.append(result)
        
        # 保存结果
        output_path = args.output.replace('.jsonl', '_sanity_check.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        print(f"\n\nSanity Check 完成！结果已保存到: {output_path}")
        print("\n结果摘要:")
        for r in results:
            print(f"  [{r['sample_name']}] mask={r['mask_type']}: {r['response'][:50]}...")
        return
    
    # ==================== 正常 POPE 推理模式 ====================
    # 加载 POPE 数据集
    print("\n[3/3] 加载 POPE 数据集...")
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
    
    # 推理
    print("\n开始推理...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for item in tqdm(questions_to_process, desc="推理中"):
            name = item['name']
            question = item['question']
            gt_cue = item['gt_cue']
            expected_answer = item['expected_answer']
            
            video_path, audio_path, face_path = get_paths_from_pope_item(item)
            
            # 检查文件是否存在
            if not os.path.exists(video_path):
                print(f"[WARNING] 视频不存在: {video_path}")
                continue
            if not os.path.exists(audio_path):
                print(f"[WARNING] 音频不存在: {audio_path}")
                continue
            if not os.path.exists(face_path):
                print(f"[WARNING] Face 不存在: {face_path}")
                continue
            
            try:
                # 只对第一个样本启用 debug
                debug_this = args.debug and (questions_to_process.index(item) == 0)
                response = run_inference(
                    chat, dataset_cls, item, face_or_frame, args.mask_type, device, debug=debug_this
                )
                
                # 提取答案
                answer = extract_answer(response)
                is_correct = (answer == expected_answer)
                
                result = {
                    "name": name,
                    "mask_type": args.mask_type,
                    "question": question,
                    "gt_cue": gt_cue,
                    "expected_answer": expected_answer,
                    "model_answer": answer,
                    "full_response": response,
                    "is_correct": is_correct,
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"[ERROR] 处理 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n推理完成！结果已保存到: {args.output}")
    
    # 计算准确率
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        if results:
            correct = sum(1 for r in results if r.get('is_correct', False))
            total = len(results)
            accuracy = correct / total * 100
            print(f"\n统计结果:")
            print(f"  总问题数: {total}")
            print(f"  正确数: {correct}")
            print(f"  准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

