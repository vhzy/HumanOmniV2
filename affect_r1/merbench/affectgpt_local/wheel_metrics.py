"""Emotion-wheel based metrics adapted from AffectGPT."""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import get_active_config
from .utils import read_csv_column, string_to_list


def _wheel_root() -> str:
    root = get_active_config().EMOTION_WHEEL_ROOT
    if not os.path.exists(root):
        raise FileNotFoundError(f"Emotion wheel assets not found at {root}")
    return root


def _read_wheel_to_map(xlsx_path: str) -> Dict[str, Dict[str, List[str]]]:
    store_map: Dict[str, Dict[str, List[str]]] = {}
    level1 = level2 = ""
    df = pd.read_excel(xlsx_path)
    for _, row in df.iterrows():
        if not pd.isna(row["level1"]):
            level1 = row["level1"]
        if not pd.isna(row["level2"]):
            level2 = row["level2"]
        if pd.isna(row["level3"]):
            continue
        level3 = row["level3"]
        l1 = str(level1).lower().strip()
        l2 = str(level2).lower().strip()
        l3 = str(level3).lower().strip()
        store_map.setdefault(l1, {}).setdefault(l2, []).append(l3)
    return store_map


def _merge_map(map1, map2):
    merged = dict(map1)
    for key, value in map2.items():
        if key in merged:
            merged[key] = sorted(set(merged[key] + value))
        else:
            merged[key] = list(value)
    return merged


@lru_cache(None)
def _format_mapping() -> Dict[str, List[str]]:
    csv_path = os.path.join(_wheel_root(), "format.csv")
    df = pd.read_csv(csv_path)
    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).lower().strip()
        formats = string_to_list(row["format"])
        formats = [fmt.lower().strip() for fmt in formats]
        mapping.setdefault(name, []).append(name)
        for fmt in formats:
            mapping.setdefault(fmt, []).append(name)
    return mapping


def _read_candidate_synonym_onerun(run_name: str) -> Dict[str, List[str]]:
    xlsx_path = os.path.join(_wheel_root(), "synonym.xlsx")
    df = pd.read_excel(xlsx_path)
    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        raw = str(row[f"word_{run_name}"]).strip().lower()
        mapping.setdefault(raw, []).append(raw)
        synonyms = string_to_list(row[f"synonym_{run_name}"])
        for syn in synonyms:
            token = syn.strip().lower()
            mapping.setdefault(token, []).append(raw)
    return mapping


@lru_cache(None)
def _raw_mapping() -> Dict[str, List[str]]:
    runs = [f"run{i}" for i in range(1, 9)]
    mapping: Dict[str, List[str]] = {}
    for run in runs:
        mapping = _merge_map(mapping, _read_candidate_synonym_onerun(run))
    return mapping


@lru_cache(None)
def _wheel_cache(wheel: str, level: str) -> Dict[str, str]:
    xlsx_path = os.path.join(_wheel_root(), f"{wheel}.xlsx")
    wheel_map_raw = _read_wheel_to_map(xlsx_path)
    mapping: Dict[str, str] = {}
    if level == "level1":
        for level1, group in wheel_map_raw.items():
            mapping[level1] = level1
            for level2, level3s in group.items():
                mapping[level2] = level1
                for level3 in level3s:
                    mapping[level3] = level1
    else:
        for level1, group in wheel_map_raw.items():
            level2_keys = sorted(group.keys())
            mapping[level1] = level2_keys[0] if level2_keys else level1
            for level2, level3s in group.items():
                mapping[level2] = level2
                for level3 in level3s:
                    mapping[level3] = level2
    return mapping


def _map_label(label: str, metric: str) -> str:
    fmt = _format_mapping()
    raw_map = _raw_mapping()
    if label not in fmt:
        return ""
    candidates = fmt[label]
    if metric.startswith("case1"):
        return sorted(candidates)[0]
    if metric.startswith("case2"):
        stage2 = []
        for cand in candidates:
            stage2.extend(raw_map.get(cand, []))
        return sorted(stage2)[0] if stage2 else ""
    if metric.startswith("case3"):
        _, wheel, level = metric.split("_")
        wheel_map = _wheel_cache(wheel, level)
        stage2 = []
        for cand in candidates:
            stage2.extend(raw_map.get(cand, []))
        for candidate in sorted(stage2):
            if candidate in wheel_map:
                return wheel_map[candidate]
        return ""
    return ""


def _normalize_words(words: Iterable[str]) -> List[str]:
    tokens = []
    for word in words:
        token = word.lower().strip()
        if token:
            tokens.append(token)
    return tokens


def calculate_openset_overlap_rate(
    *,
    name2gt: Dict[str, str],
    name2pred: Dict[str, str],
    metric: str,
    inter_print: bool = True,
) -> Tuple[float, float]:
    precision, recall = [], []
    processed = 0
    for name, gt_raw in name2gt.items():
        if name not in name2pred:
            continue
        gt = _normalize_words(string_to_list(gt_raw))
        pred = _normalize_words(string_to_list(name2pred[name]))
        gt = [mapped for mapped in (_map_label(label, metric) for label in gt) if mapped]
        pred = [mapped for mapped in (_map_label(label, metric) for label in pred) if mapped]
        if not gt:
            continue
        processed += 1
        gt_set, pred_set = set(gt), set(pred)
        if not pred_set:
            precision.append(0.0)
            recall.append(0.0)
        else:
            precision.append(len(gt_set & pred_set) / len(pred_set))
            recall.append(len(gt_set & pred_set) / len(gt_set))
    if processed == 0:
        return 0.0, 0.0
    if inter_print:
        print(f"process number (after filter): {processed}")
    return float(np.mean(precision)), float(np.mean(recall))


def wheel_metric_calculation(
    *,
    name2gt: Dict[str, str],
    name2pred: Dict[str, str],
    inter_print: bool = True,
    level: str = "level1",
) -> List[float]:
    if level == "level1":
        metrics = [f"case3_wheel{i}_level1" for i in range(1, 6)]
    else:
        metrics = [f"case3_wheel{i}_level2" for i in range(1, 6)]
    scores = []
    for metric in metrics:
        prec, rec = calculate_openset_overlap_rate(
            name2gt=name2gt,
            name2pred=name2pred,
            metric=metric,
            inter_print=inter_print,
        )
        if prec + rec == 0:
            fscore = 0.0
        else:
            fscore = 2 * (prec * rec) / (prec + rec)
        scores.append([fscore, prec, rec])
    return np.mean(scores, axis=0).tolist()


def _candidate_labels(name2gt: Dict[str, str]) -> List[str]:
    labels = []
    for value in name2gt.values():
        labels.extend(string_to_list(value))
    return labels


def calculate_openset_onehot_hitrate(
    *,
    name2gt: Dict[str, str],
    name2pred: Dict[str, str],
    metric: str,
    inter_print: bool = True,
) -> Tuple[float, float]:
    candidates = _normalize_words(_candidate_labels(name2gt))
    candidate_set = set(_map_label(label, metric) for label in candidates)
    candidate_set.discard("")
    hitrates, mscores = [], []
    processed = 0
    for name, gt_raw in name2gt.items():
        if name not in name2pred:
            continue
        gt = _normalize_words(string_to_list(gt_raw))
        gt = [mapped for mapped in (_map_label(label, metric) for label in gt) if mapped]
        if not gt:
            continue
        pred = _normalize_words(string_to_list(name2pred[name]))
        pred = [mapped for mapped in (_map_label(label, metric) for label in pred) if mapped]
        processed += 1
        hits = len(set(pred) & set(gt))
        hitrates.append(hits)
        overlap = len(set(pred) & candidate_set)
        mscores.append(0.0 if overlap == 0 else hits / overlap)
    if processed == 0:
        return 0.0, 0.0
    if inter_print:
        print(f"after filter sample number: {processed}")
    return float(np.mean(hitrates)), float(np.mean(mscores))


def hitrate_metric_calculation(
    *,
    name2gt: Dict[str, str],
    name2pred: Dict[str, str],
    inter_print: bool = True,
) -> Tuple[float, float]:
    metrics = [f"case3_wheel{i}_level1" for i in range(1, 6)]
    scores = []
    for metric in metrics:
        scores.append(
            calculate_openset_onehot_hitrate(
                name2gt=name2gt,
                name2pred=name2pred,
                metric=metric,
                inter_print=inter_print,
            )
        )
    return tuple(np.mean(scores, axis=0))

