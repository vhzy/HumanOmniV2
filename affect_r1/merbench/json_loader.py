import json
from typing import Dict, Iterable, Iterator, Tuple


def iter_prediction_rows(jsonl_path: str) -> Iterator[Dict]:
    """Yield parsed rows from a JSONL prediction file."""
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_name_index(jsonl_path: str) -> Dict[str, Dict]:
    """Return a mapping from sample name to the latest prediction row."""
    name2row: Dict[str, Dict] = {}
    for row in iter_prediction_rows(jsonl_path):
        name = row.get("name")
        if not name:
            continue
        name2row[name] = row
    return name2row


def collect_reason_and_answer(jsonl_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return dictionaries for name -> reason text, name -> open-vocabulary answer."""
    name2reason, name2answer = {}, {}
    for row in iter_prediction_rows(jsonl_path):
        name = row.get("name")
        if not name:
            continue
        think = (row.get("think") or "").strip()
        answer = (row.get("answer") or "").strip()
        reason_text = think
        if answer:
            if reason_text:
                reason_text = f"{reason_text}\nFinal answer: {answer}"
            else:
                reason_text = answer
        name2reason[name] = reason_text
        name2answer[name] = answer
    return name2reason, name2answer

