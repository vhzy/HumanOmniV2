#!/usr/bin/env python3
import argparse
import json

def _is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return len(x.strip()) == 0
    if isinstance(x, (list, tuple, dict, set)):
        return len(x) == 0
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_with_extracted_clues_fixed.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_with_extracted_clues_filtered.jsonl",
    )
    args = ap.parse_args()

    total = 0
    empty_any = 0
    empty_visual = 0
    empty_audio = 0
    kept = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            clues = obj.get("extracted_clues") or {}
            visual = clues.get("visual_clues")
            audio = clues.get("audio_clues")

            v_empty = _is_empty(visual)
            a_empty = _is_empty(audio)

            if v_empty:
                empty_visual += 1
            if a_empty:
                empty_audio += 1

            if v_empty or a_empty:
                empty_any += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"total: {total}")
    print(f"empty_visual: {empty_visual}")
    print(f"empty_audio:  {empty_audio}")
    print(f"empty_any:    {empty_any}")
    print(f"kept:         {kept}")


if __name__ == "__main__":
    main()
