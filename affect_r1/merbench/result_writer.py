import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_tag(text: str, pattern: re.Pattern) -> Optional[str]:
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_think_answer(raw_text: str) -> Dict[str, Any]:
    """Parse `<think>` and `<answer>` segments from a completion string."""
    raw_text = raw_text or ""
    think = _extract_tag(raw_text, THINK_PATTERN)
    answer = _extract_tag(raw_text, ANSWER_PATTERN)
    valid = think is not None and answer is not None
    if think is None:
        think = ""
    if answer is None:
        # fall back to stripped text (best effort)
        answer = raw_text.strip()
    return {"think": think, "answer": answer, "valid_format": valid}


@dataclass
class MerBenchResultWriter:
    """
    Utility class that stores inference outputs into JSONL files compatible
    with the MER-UniBench evaluation pipeline.
    """

    output_root: str
    dataset: str
    run_name: str
    checkpoint_name: str
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        dataset_slug = self.dataset.lower()
        self.save_dir = os.path.join(
            self.output_root,
            f"results-{dataset_slug}",
            self.run_name,
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.save_dir, f"{self.checkpoint_name}.jsonl")
        # Open in append mode so that retries will continue the same file.
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")

    def close(self):
        if getattr(self, "_fh", None):
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def log_sample(
        self,
        name: str,
        response_text: str,
        subtitle: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Append one sample to the JSONL file."""
        parsed = parse_think_answer(response_text)
        record = {
            "name": name,
            "dataset": self.dataset,
            "run_name": self.run_name,
            "checkpoint": self.checkpoint_name,
            "timestamp": time.time(),
            "raw_response": response_text,
            "think": parsed["think"],
            "answer": parsed["answer"],
            "has_valid_format": parsed["valid_format"],
            "subtitle": subtitle or "",
            "prompt": prompt or "",
            "metadata": metadata or {},
        }
        if self.extra_metadata:
            record["metadata"] = {**self.extra_metadata, **record["metadata"]}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

