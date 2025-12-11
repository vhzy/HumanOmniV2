"""Reusable helpers mirroring AffectGPT's toolkit utilities."""

from __future__ import annotations

import math
import os
import re
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def string_to_list(value):
    """Convert serialized list strings to python lists (compatible w/ AffectGPT)."""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text[0] == "[":
        text = text[1:]
    if text and text[-1] == "]":
        text = text[:-1]
    items = [
        item.strip()
        for item in re.split(r"[\"',]", text)
        if item.strip() not in {"", ","}
    ]
    return items


def split_list_into_batch(items: Sequence[T], batchsize: int | None = None, split_num: int | None = None) -> List[Sequence[T]]:
    """Chunk a sequence either by desired batchsize or number of splits."""
    if batchsize is None and split_num is None:
        raise ValueError("Provide either batchsize or split_num.")
    if split_num is None:
        split_num = math.ceil(len(items) / max(batchsize, 1))
    each = math.ceil(len(items) / max(split_num, 1))
    batches = []
    for idx in range(split_num):
        chunk = items[idx * each : (idx + 1) * each]
        if chunk:
            batches.append(chunk)
    return batches


def iter_batches(items: Sequence[T], batchsize: int) -> Iterator[Sequence[T]]:
    for idx in range(0, len(items), batchsize):
        yield items[idx : idx + batchsize]


def read_csv_column(csv_path: str, column: str) -> list[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    values = []
    if column not in df.columns:
        return ["" for _ in range(len(df))]
    for _, row in df.iterrows():
        val = row[column]
        if pd.isna(val):
            values.append("")
        else:
            values.append(str(val))
    return values

