#!/usr/bin/env python3
"""
将 AffectGPT 的 NPZ 输出转换为 JSONL 格式，以便 check_hallucination.py 使用

AffectGPT NPZ 格式：
- name2reason: dict, {sample_name: reason_text}

输出 JSONL 格式（每行一个 JSON 对象）：
{
    "name": "sample_00000000",
    "think": "reason text...",  # 直接使用 reason 作为 think
    "metadata": {
        "counterfactual_config": {
            "mask_modality": "none"  # baseline 模式，不 mask 任何模态
        }
    }
}

使用示例：
python convert_affectgpt_npz_to_jsonl.py \
    --input /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/ovmerd_description/checkpoint_000030_loss_0.602.npz \
    --output /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/ovmerd_description/checkpoint_000030_loss_0.602.jsonl
"""

import argparse
import json
import numpy as np
import os
from tqdm import tqdm


def convert_npz_to_jsonl(input_path: str, output_path: str, mask_modality: str = "none"):
    """
    将 AffectGPT NPZ 文件转换为 JSONL 格式
    
    Args:
        input_path: NPZ 文件路径
        output_path: 输出 JSONL 文件路径
        mask_modality: mask 模态类型 ("none", "visual", "audio")
    """
    print(f"[INFO] Loading NPZ file: {input_path}")
    
    with np.load(input_path, allow_pickle=True) as data:
        print(f"[INFO] NPZ keys: {list(data.keys())}")
        name2reason = data["name2reason"].item()  # 取出 dict
        print(f"[INFO] Total samples: {len(name2reason)}")
    
    # 转换为 JSONL
    print(f"[INFO] Converting to JSONL: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for name, reason in tqdm(name2reason.items(), desc="Converting"):
            record = {
                "name": name,
                "think": reason,  # AffectGPT 的 reason 直接作为 think 内容
                "metadata": {
                    "counterfactual_config": {
                        "mask_modality": mask_modality
                    }
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[INFO] Done! Converted {len(name2reason)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert AffectGPT NPZ to JSONL for check_hallucination.py")
    parser.add_argument("--input", "-i", required=True, help="Input NPZ file path")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL file path (default: same as input with .jsonl extension)")
    parser.add_argument("--mask-modality", default="none", choices=["none", "visual", "audio"],
                        help="Mask modality type for hallucination detection (default: none = baseline mode)")
    
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        args.output = args.input.replace(".npz", ".jsonl")
    
    convert_npz_to_jsonl(args.input, args.output, args.mask_modality)


if __name__ == "__main__":
    main()

