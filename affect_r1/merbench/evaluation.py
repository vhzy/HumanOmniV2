import argparse
import json
import os
import re
import sys
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .affect_config import create_local_config
from .affectgpt_local import build_local_modules
from .json_loader import collect_reason_and_answer


def _namespace_to_module(ns) -> types.ModuleType:
    module = types.ModuleType("config")
    for key, value in vars(ns).items():
        setattr(module, key, value)
    return module


def load_config(args):
    local_cfg = create_local_config(args.dataset_root)
    cfg_module = _namespace_to_module(local_cfg)
    sys.modules["config"] = cfg_module
    return cfg_module


def dataset_flag(name: str) -> str:
    if name in {"MER2023", "MER2024", "MELD", "IEMOCAPFOUR"}:
        return "discrete"
    if name in {"CMUMOSI", "CMUMOSEI", "SIMS", "SIMSV2"}:
        return "dimension"
    if name in {"MER2025OV", "OVMERDPLUS"}:
        return "ovlabel"
    raise ValueError(f"Unsupported dataset: {name}")


def get_emo_maps(dataset_cls) -> Tuple[Dict, Dict]:
    if hasattr(dataset_cls, "get_emo2idx_idx2emo"):
        return dataset_cls.get_emo2idx_idx2emo()
    return {}, {}


def parse_answer_tokens(answer_text: str):
    stripped = (answer_text or "").strip()
    if not stripped:
        return []
    stripped = stripped.replace("[", "").replace("]", "")
    tokens = [part.strip().lower() for part in re.split(r"[,\n;，；]+", stripped)]
    return [tok for tok in tokens if tok]


def tokens_to_list_string(tokens):
    if not tokens:
        return "[]"
    return f"[{', '.join(tokens)}]"


def parse_sentiment_response(resp: str) -> str:
    """Normalize LLM sentiment outputs to {positive, negative, neutral}."""
    if not resp:
        return "neutral"
    cleaned = resp.strip().lower()
    strip_prefixes = [
        "output:",
        "output：",
        "sentiment:",
        "sentiment：",
        "the sentiment is",
        "the most likely sentiment is",
        "overall sentiment:",
        "overall sentiment：",
        "therefore, the sentiment is",
        "response:",
        "response：",
        "answer:",
        "answer：",
        "final output:",
        "final output：",
        "result:",
        "result：",
    ]
    for prefix in strip_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip("[]()\"' ")
    match = re.search(r"\b(positive|negative|neutral)\b", cleaned)
    if match:
        return match.group(1)
    print(f"[WARN] Unrecognized sentiment response, defaulting to neutral: {resp[:120]}")
    return "neutral"


def answers_to_openset_map(name2answer: Dict[str, str]):
    return {
        name: tokens_to_list_string(parse_answer_tokens(answer))
        for name, answer in name2answer.items()
    }


def load_predictions(jsonl_path: str):
    name2reason, name2answer = collect_reason_and_answer(jsonl_path)
    return name2reason, name2answer


@contextmanager
def duplicate_console(log_path: str):
    if not log_path:
        yield
        return
    log_path = os.path.abspath(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
            self.encoding = getattr(streams[0], "encoding", "utf-8")

        def write(self, data):
            for stream in self._streams:
                stream.write(data)
            return len(data)

        def flush(self):
            for stream in self._streams:
                stream.flush()

        def isatty(self):
            return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

        def __getattr__(self, attr):
            return getattr(self._streams[0], attr)

    stdout_backup, stderr_backup = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    try:
        yield
    finally:
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        log_fh.close()


def align_with_groundtruth(name2gt: Dict, name2reason: Dict, name2answer: Dict):
    available_names = sorted(set(name2gt.keys()) & set(name2reason.keys()))
    filtered_gt = {name: name2gt[name] for name in available_names}
    filtered_reason = {name: name2reason[name] for name in available_names}
    filtered_answer = {name: name2answer.get(name, "") for name in available_names}
    return filtered_gt, filtered_reason, filtered_answer


def evaluate_discrete(
    name2gt,
    name2reason,
    modules,
    llm_tuple,
    *,
    use_answer_openset: bool,
    answer_openset: Dict[str, str],
):
    if use_answer_openset:
        name2pred = {name: answer_openset.get(name, "[]") for name in name2gt}
    else:
        extract_openset = modules["extract_openset"]
        names, responses = extract_openset(
            name2reason=name2reason,
            llm=llm_tuple[0],
            tokenizer=llm_tuple[1],
            sampling_params=llm_tuple[2],
        )
        name2pred = dict(zip(names, responses))
    hitrate, mscore = modules["hitrate_metric"](
        name2gt=name2gt,
        name2pred=name2pred,
        inter_print=True,
    )
    return {"hitrate": hitrate, "mscore": mscore}


def evaluate_dimension(
    name2gt,
    name2reason,
    modules,
    llm_tuple,
    *,
    use_answer_openset: bool,
    answer_openset: Dict[str, str],
    jsonl_path: str = None,
):
    if use_answer_openset:
        name2openset = {name: answer_openset.get(name, "[]") for name in name2reason}
    else:
        extract_openset = modules["extract_openset"]
        openset_names, openset_responses = extract_openset(
            name2reason=name2reason,
            llm=llm_tuple[0],
            tokenizer=llm_tuple[1],
            sampling_params=llm_tuple[2],
        )
        name2openset = dict(zip(openset_names, openset_responses))
    
    # 尝试加载缓存的 sentiment 结果
    sentiment_cache_path = None
    if jsonl_path:
        base_path = jsonl_path.rsplit(".", 1)[0]  # 去掉 .jsonl 后缀
        sentiment_cache_path = f"{base_path}-sentiment.json"
    
    name2sentiment = None
    if sentiment_cache_path and os.path.exists(sentiment_cache_path):
        print(f"[INFO] Loading cached sentiment from: {sentiment_cache_path}")
        with open(sentiment_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            name2sentiment = {
                name: parse_sentiment_response(value)
                for name, value in cached.items()
            }
    
    if name2sentiment is None:
        # 调用 LLM 生成 sentiment
        sent_names, sentiments = modules["openset_to_sentiment"](
            name2openset=name2openset,
            llm=llm_tuple[0],
            tokenizer=llm_tuple[1],
            sampling_params=llm_tuple[2],
        )
        parsed_sentiments = [parse_sentiment_response(item) for item in sentiments]
        name2sentiment = dict(zip(sent_names, parsed_sentiments))
        
        # 保存到缓存文件
        if sentiment_cache_path:
            print(f"[INFO] Saving sentiment cache to: {sentiment_cache_path}")
            with open(sentiment_cache_path, "w", encoding="utf-8") as f:
                json.dump(name2sentiment, f, ensure_ascii=False, indent=2)

    ordered_names = [name for name in name2reason.keys() if name in name2sentiment]
    val_labels = np.array([name2gt[name] for name in ordered_names])
    val_preds = []
    for name in ordered_names:
        label = name2sentiment[name].strip().lower()
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
    return {"fscore": fscore, "accuracy": acc}


def evaluate_ovlabel(name2gt, name2answer, modules):
    name2pred = answers_to_openset_map(name2answer)
    fscore, precision, recall = modules["wheel_metric"](
        name2gt=name2gt,
        name2pred=name2pred,
        inter_print=True,
    )
    return {"fscore": fscore, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MER-UniBench JSONL outputs.")
    parser.add_argument("--jsonl-path", required=False, help="Path to checkpoint jsonl file.")
    parser.add_argument(
        "--results-root",
        required=False,
        help="Directory like output/results-mer2024/<run_name>.",
    )
    parser.add_argument("--checkpoint-name", required=False, help="Checkpoint file stem, e.g. checkpoint_000030.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. MER2024.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing MER datasets.")
    parser.add_argument("--llm-name", default="Qwen25", help="Key inside AffectGPT config.PATH_TO_LLM.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls when they are not required.")
    parser.add_argument(
        "--use-answer-openset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use parsed <answer> tokens as openset input (default: True).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to save console output logs.",
    )
    args = parser.parse_args()

    log_ctx = duplicate_console(args.log_file) if args.log_file else nullcontext()
    with log_ctx:
        load_config(args)
        dataset_name = args.dataset.strip().upper()
        modules = build_local_modules()

        if args.jsonl_path:
            jsonl_path = args.jsonl_path
        else:
            if not (args.results_root and args.checkpoint_name):
                raise ValueError("Provide either --jsonl-path or both --results-root and --checkpoint-name.")
            jsonl_path = os.path.join(args.results_root, f"{args.checkpoint_name}.jsonl")

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Prediction file not found: {jsonl_path}")

        dataset_cls = modules["dataset_map"][dataset_name]()
        name2gt = dataset_cls.get_test_name2gt()
        flag = dataset_flag(dataset_name)
        if flag == "discrete":
            _, idx2emo = get_emo_maps(dataset_cls)
            for name in name2gt:
                if not isinstance(name2gt[name], str):
                    name2gt[name] = idx2emo[name2gt[name]]

        name2reason, name2answer = load_predictions(jsonl_path)
        name2gt, name2reason, name2answer = align_with_groundtruth(name2gt, name2reason, name2answer)
        if not name2gt:
            raise RuntimeError("No overlapping samples between predictions and ground-truth.")
        answer_openset = answers_to_openset_map(name2answer)

        needs_llm = flag == "dimension" or (flag == "discrete" and not args.use_answer_openset)
        llm_tuple = None
        if needs_llm:
            if args.skip_llm:
                raise ValueError("Cannot skip LLM when extraction or sentiment conversion is required.")
            llm_tuple = modules["load_llm"](args.llm_name)

        results = {}
        if flag == "ovlabel":
            results = evaluate_ovlabel(name2gt, name2answer, modules)
        else:
            if flag == "discrete":
                results = evaluate_discrete(
                    name2gt,
                    name2reason,
                    modules,
                    llm_tuple,
                    use_answer_openset=args.use_answer_openset,
                    answer_openset=answer_openset,
                )
            elif flag == "dimension":
                results = evaluate_dimension(
                    name2gt,
                    name2reason,
                    modules,
                    llm_tuple,
                    use_answer_openset=args.use_answer_openset,
                    answer_openset=answer_openset,
                    jsonl_path=jsonl_path,
                )

        print(f"[{dataset_name}] {results}")


if __name__ == "__main__":
    main()

