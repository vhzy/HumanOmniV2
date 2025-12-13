import argparse
import json
import os
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对齐官方用法：取有效 token 的最后一位（左填充则末位）。"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_len]


def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def encode_phrases(
    phrases: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    if len(phrases) == 0:
        return torch.empty(0, model.config.hidden_size)
    inputs = tokenizer(
        phrases,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu()


def build_rubric_db(
    data_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    db: Dict[str, Dict[str, torch.Tensor]] = {}
    for sample in tqdm(load_jsonl(data_path), desc="processing"):
        path = sample.get("path")
        clues = sample.get("extracted_clues") or {}
        if not path:
            continue

        visual = clues.get("visual_clues") or []
        audio = clues.get("audio_clues") or []
        logic = clues.get("reasoning_emotions") or []

        # 忽略完全空的样本（正常情况下已过滤）
        if len(visual) == 0 and len(audio) == 0 and len(logic) == 0:
            continue

        db[path] = {
            "visual": encode_phrases(visual, tokenizer, model, device),
            "audio": encode_phrases(audio, tokenizer, model, device),
            "logic": encode_phrases(logic, tokenizer, model, device),
        }
    return db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_with_extracted_clues.jsonl",
        help="包含 extracted_clues 的 RL 数据 jsonl",
    )
    parser.add_argument(
        "--embedder_path",
        type=str,
        default="/mnt/afs/hanzhiyuan/huggingface/Qwen3-Embedding-0.6B",
        help="Qwen3-Embedding-0.6B 路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_rubric_embeddings2.pt",
        help="保存 torch.save 的输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="编码设备，默认自动选择 cuda 可用则用 cuda，否则 cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.embedder_path)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(args.embedder_path).to(device)
    model.eval()

    db = build_rubric_db(args.data_path, tokenizer, model, device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(db, args.output_path)
    print(f"Saved rubric embeddings to {args.output_path}, samples={len(db)}")


if __name__ == "__main__":
    main()
