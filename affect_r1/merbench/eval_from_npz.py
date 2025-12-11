#!/usr/bin/env python
'''
cd /mnt/afs/hanzhiyuan/code/HumanOmniV2
python -m affect_r1.merbench.eval_from_npz \
      --dataset MER2023 \
      --dataset-root /mnt/afs/hanzhiyuan/datasets \
      --checkpoint-base /mnt/afs/hanzhiyuan/code/AffectGPT/AffectGPT/output/results-mer2023/emercoarse.../checkpoint_000050_loss_0.527
'''
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .affect_config import create_local_config
from .affectgpt_local import build_local_modules
from .evaluation import dataset_flag, get_emo_maps, tokens_to_list_string


def _namespace_to_module(ns) -> object:
    import types
    import sys as _sys

    module = types.ModuleType("config")
    for key, value in vars(ns).items():
        setattr(module, key, value)
    _sys.modules["config"] = module
    return module


def load_config(dataset_root: str):
    cfg = create_local_config(dataset_root)
    _namespace_to_module(cfg)
    return cfg


def load_reason_npz(path: str) -> Dict[str, str]:
    if not path:
        return {}
    data = np.load(path, allow_pickle=True)
    if "name2reason" in data:
        payload = data["name2reason"].tolist()
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    filenames = data.get("filenames")
    fileitems = data.get("fileitems")
    if filenames is None or fileitems is None:
        raise ValueError(f"Unsupported NPZ structure for reasons: {path}")
    return {str(name): str(item) for name, item in zip(filenames.tolist(), fileitems.tolist())}


def load_openset_npz(path: str) -> Dict[str, List[str]]:
    data = np.load(path, allow_pickle=True)
    filenames = data.get("filenames")
    fileitems = data.get("fileitems")
    if filenames is None or fileitems is None:
        raise ValueError(f"Unsupported NPZ structure for openset: {path}")
    result = {}
    for name, item in zip(filenames.tolist(), fileitems.tolist()):
        if isinstance(item, str):
            tokens = [tok.strip() for tok in item.strip("[]").split(",") if tok.strip()]
        else:
            tokens = []
            for tok in item:
                if tok is None:
                    continue
                tokens.append(str(tok).strip())
        result[str(name)] = [tok for tok in tokens if tok]
    return result


def load_sentiment_npz(path: str) -> Dict[str, str]:
    if not path:
        return {}
    data = np.load(path, allow_pickle=True)
    filenames = data.get("filenames")
    fileitems = data.get("fileitems")
    if filenames is None or fileitems is None:
        raise ValueError(f"Unsupported NPZ structure for sentiment: {path}")
    return {str(name): str(item).strip().lower() for name, item in zip(filenames.tolist(), fileitems.tolist())}


def align_samples(name2gt: Dict[str, str], *pred_dicts: Dict[str, object]) -> Tuple[Dict[str, str], List[str]]:
    candidate_sets = [set(d.keys()) for d in pred_dicts if d]
    if not candidate_sets:
        return name2gt, sorted(name2gt.keys())
    valid_names = set(name2gt.keys())
    for cand in candidate_sets:
        valid_names &= cand
    ordered_names = sorted(valid_names)
    filtered_gt = {name: name2gt[name] for name in ordered_names}
    return filtered_gt, ordered_names


def evaluate_discrete_with_openset(name2gt: Dict[str, str], openset_dict: Dict[str, List[str]], modules):
    name2pred = {}
    for name in name2gt:
        tokens = openset_dict.get(name, [])
        name2pred[name] = tokens_to_list_string(tokens)
    hitrate, mscore = modules["hitrate_metric"](
        name2gt=name2gt,
        name2pred=name2pred,
        inter_print=True,
    )
    return {"hitrate": float(hitrate), "mscore": float(mscore)}


def evaluate_ov_with_openset(name2gt: Dict[str, str], openset_dict: Dict[str, List[str]], modules):
    name2pred = {}
    for name in name2gt:
        tokens = openset_dict.get(name, [])
        name2pred[name] = tokens_to_list_string(tokens)
    fscore, precision, recall = modules["wheel_metric"](
        name2gt=name2gt,
        name2pred=name2pred,
        inter_print=True,
    )
    return {
        "fscore": float(fscore),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_dimension_with_sentiment(name2gt: Dict[str, float], name2sentiment: Dict[str, str]):
    ordered_names = [name for name in name2gt if name in name2sentiment]
    if not ordered_names:
        raise RuntimeError("No overlapping samples between predictions and ground-truth.")
    val_labels = np.array([name2gt[name] for name in ordered_names])
    val_preds = []
    for name in ordered_names:
        label = name2sentiment[name]
        if label == "positive":
            val_preds.append(1)
        elif label == "negative":
            val_preds.append(-1)
        else:
            val_preds.append(0)
    val_preds = np.array(val_preds)
    non_zero_idx = np.array([i for i, v in enumerate(val_labels) if v != 0])
    if len(non_zero_idx) == 0:
        return {"fscore": 0.0, "accuracy": 0.0}
    fscore = f1_score(
        (val_labels[non_zero_idx] > 0),
        (val_preds[non_zero_idx] > 0),
        average="weighted",
    )
    acc = accuracy_score(
        (val_labels[non_zero_idx] > 0),
        (val_preds[non_zero_idx] > 0),
    )
    return {"fscore": float(fscore), "accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MER-UniBench NPZ outputs without re-running LLM extraction.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. MER2024.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing MER datasets.")
    parser.add_argument("--checkpoint-base", default=None, help="Path prefix to checkpoint (without suffix).")
    parser.add_argument("--reason-npz", default=None, help="Explicit path to *.npz storing name2reason.")
    parser.add_argument("--openset-npz", default=None, help="Explicit path to *-openset.npz.")
    parser.add_argument("--sentiment-npz", default=None, help="Explicit path to *-openset-sentiment.npz.")
    args = parser.parse_args()

    dataset_name = args.dataset.strip().upper()

    def resolve(path_suffix: str):
        if args.checkpoint_base:
            candidate = f"{args.checkpoint_base}{path_suffix}"
            if os.path.exists(candidate):
                return candidate
        return None

    reason_npz = args.reason_npz or resolve(".npz")
    openset_npz = args.openset_npz or resolve("-openset.npz")
    sentiment_npz = args.sentiment_npz or resolve("-openset-sentiment.npz")

    load_config(args.dataset_root)
    modules = build_local_modules()
    dataset_cls = modules["dataset_map"][dataset_name]()
    name2gt = dataset_cls.get_test_name2gt()

    flag = dataset_flag(dataset_name)
    if flag == "discrete":
        _, idx2emo = get_emo_maps(dataset_cls)
        for name in name2gt:
            if not isinstance(name2gt[name], str):
                name2gt[name] = idx2emo[name2gt[name]]

    reason_dict = load_reason_npz(reason_npz) if reason_npz else {}
    openset_dict = load_openset_npz(openset_npz) if openset_npz else {}
    sentiment_dict = load_sentiment_npz(sentiment_npz) if sentiment_npz else {}

    if flag in {"discrete", "ovlabel"} and not openset_dict:
        raise ValueError("openset npz is required for discrete and ovlabel datasets.")
    if flag == "dimension" and not sentiment_dict:
        raise ValueError("sentiment npz is required for dimension datasets.")

    name2gt, ordered_names = align_samples(name2gt, reason_dict, openset_dict, sentiment_dict)
    if not name2gt:
        raise RuntimeError("No overlapping samples between predictions and ground-truth.")

    if flag == "discrete":
        results = evaluate_discrete_with_openset(name2gt, openset_dict, modules)
    elif flag == "dimension":
        results = evaluate_dimension_with_sentiment(name2gt, sentiment_dict)
    elif flag == "ovlabel":
        results = evaluate_ov_with_openset(name2gt, openset_dict, modules)
    else:
        raise ValueError(f"Unsupported dataset flag: {flag}")

    print(f"[{dataset_name}] {results}")


if __name__ == "__main__":
    main()

