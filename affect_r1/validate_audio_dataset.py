import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional

# 优先使用 torchaudio，若缺失则回落到 soundfile
try:
    import torchaudio  # type: ignore
except Exception:  # pragma: no cover - 环境缺失时走回退
    torchaudio = None
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        sf = None  # type: ignore
else:
    sf = None  # type: ignore


def read_jsonl(path: str, max_samples: Optional[int]) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


def probe_audio(path: str) -> Dict:
    if not os.path.isfile(path):
        return {"status": "missing", "path": path}

    if torchaudio is not None:
        try:
            info = torchaudio.info(path)
            frames = info.num_frames
            sr = info.sample_rate
            if frames <= 0 or sr <= 0:
                return {"status": "invalid", "path": path, "detail": "non-positive frames or sample_rate"}
            duration = frames / float(sr)
            return {"status": "ok", "path": path, "frames": frames, "sample_rate": sr, "duration": duration}
        except Exception as e:  # pragma: no cover - 调试用信息
            return {"status": "error", "path": path, "detail": str(e)}

    if sf is not None:
        try:
            with sf.SoundFile(path) as snd:
                frames = len(snd)
                sr = snd.samplerate
            if frames <= 0 or sr <= 0:
                return {"status": "invalid", "path": path, "detail": "non-positive frames or sample_rate"}
            duration = frames / float(sr)
            return {"status": "ok", "path": path, "frames": frames, "sample_rate": sr, "duration": duration}
        except Exception as e:  # pragma: no cover
            return {"status": "error", "path": path, "detail": str(e)}

    return {"status": "error", "path": path, "detail": "no audio backend available"}


def collect_audio_paths(records: Iterator[Dict]) -> List[str]:
    paths: List[str] = []
    for rec in records:
        audio_path = rec.get("audio_path")
        if isinstance(audio_path, str):
            paths.append(audio_path)
    return paths


def validate_dataset(jsonl_path: str, max_samples: Optional[int], num_workers: int) -> Dict:
    records = read_jsonl(jsonl_path, max_samples)
    audio_paths = collect_audio_paths(records)
    counts = Counter(audio_paths)
    duplicate_entries = [{"path": p, "count": c} for p, c in counts.items() if c > 1]
    unique_paths = list(dict.fromkeys(audio_paths))  # 去重并保序

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(probe_audio, p): p for p in unique_paths}
        for future in as_completed(future_to_path):
            results.append(future.result())

    summary = {"total_jsonl_rows": len(audio_paths), "unique_audio_files": len(unique_paths), "stats": {}}
    for r in results:
        summary["stats"].setdefault(r["status"], 0)
        summary["stats"][r["status"]] += 1
    summary["duplicates_count"] = len(duplicate_entries)
    summary["duplicate_examples"] = sorted(duplicate_entries, key=lambda x: -x["count"])[:20]

    def collect_bad(status: str, limit: int = 20) -> List[Dict]:
        return [r for r in results if r["status"] == status][:limit]

    summary["missing_examples"] = collect_bad("missing")
    summary["error_examples"] = collect_bad("error")
    summary["invalid_examples"] = collect_bad("invalid")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证音频数据可读性与基本属性")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl/rl_rest_5000_with_rl.jsonl",
        help="包含 audio_path 字段的 jsonl 路径",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="仅检查前 N 条（默认全量）")
    parser.add_argument("--num-workers", type=int, default=16, help="并发探测线程数")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = validate_dataset(args.jsonl, args.max_samples, args.num_workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

