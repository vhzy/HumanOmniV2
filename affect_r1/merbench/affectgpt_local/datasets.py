"""Lightweight dataset adapters mirroring AffectGPT interfaces."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .common import get_active_config
from .utils import read_csv_column, string_to_list


class BaseEvalDataset:
    dataset: str = ""

    def __init__(self):
        self.cfg = get_active_config()

    def _npz_dict(self, key_candidates: tuple[str, ...]) -> dict:
        label_path = self.cfg.PATH_TO_LABEL[self.dataset]
        if not label_path or not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for {self.dataset}: {label_path}")
        data = np.load(label_path, allow_pickle=True)
        for key in key_candidates:
            if key in data:
                return data[key].tolist()
        raise KeyError(f"Keys {key_candidates} not found in {label_path}")

    def get_test_name2gt(self) -> Dict[str, str]:
        raise NotImplementedError

    def get_emo2idx_idx2emo(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return {}, {}


class MER202XDataset(BaseEvalDataset):
    dataset: str
    test_key = ("test1_corpus",)
    train_key = ("train_corpus",)

    def get_test_name2gt(self):
        corpus = self._npz_dict(self.test_key)
        return {name: corpus[name]["emo"] for name in corpus}

    def get_emo2idx_idx2emo(self):
        try:
            corpus = self._npz_dict(self.train_key)
        except KeyError:
            corpus = self._npz_dict(self.test_key)
        emos = sorted({value["emo"] for value in corpus.values()})
        emo2idx = {emo: idx for idx, emo in enumerate(emos)}
        idx2emo = {idx: emo for emo, idx in emo2idx.items()}
        return emo2idx, idx2emo


class MER2023Dataset(MER202XDataset):
    dataset = "MER2023"


class MER2024Dataset(MER202XDataset):
    dataset = "MER2024"


class MELDDataset(BaseEvalDataset):
    dataset = "MELD"
    EMOS = ["anger", "joy", "sadness", "neutral", "disgust", "fear", "surprise"]

    def get_test_name2gt(self):
        corpus = self._npz_dict(("test_corpus",))
        return {name: corpus[name]["emo"] for name in corpus}

    def get_emo2idx_idx2emo(self):
        emo2idx = {emo: idx for idx, emo in enumerate(self.EMOS)}
        idx2emo = {idx: emo for emo, idx in emo2idx.items()}
        return emo2idx, idx2emo


class IEMOCAPFourDataset(BaseEvalDataset):
    dataset = "IEMOCAPFour"
    EMOS = ["happy", "sad", "neutral", "anger"]

    def get_test_name2gt(self):
        corpus = self._npz_dict(("whole_corpus",))
        idx2emo = {idx: emo for idx, emo in enumerate(self.EMOS)}
        name2gt = {}
        for name, payload in corpus.items():
            if len(name) > 4 and name[4] == "5":  # Session 5 split
                name2gt[name] = idx2emo[payload["emo"]]
        return name2gt

    def get_emo2idx_idx2emo(self):
        emo2idx = {emo: idx for idx, emo in enumerate(self.EMOS)}
        idx2emo = {idx: emo for emo, idx in emo2idx.items()}
        return emo2idx, idx2emo


class NPZValenceDataset(BaseEvalDataset):
    dataset: str

    def get_test_name2gt(self):
        corpus = self._npz_dict(("test_corpus",))
        return {name: float(corpus[name]["val"]) for name in corpus}


class SIMSDataset(NPZValenceDataset):
    dataset = "SIMS"


class SIMSv2Dataset(NPZValenceDataset):
    dataset = "SIMSv2"


class CMUMOSIDataset(NPZValenceDataset):
    dataset = "CMUMOSI"


class CMUMOSEIDataset(NPZValenceDataset):
    dataset = "CMUMOSEI"


class MER2025OVDataset(BaseEvalDataset):
    dataset = "MER2025OV"

    def get_test_name2gt(self):
        csv_path = self.cfg.PATH_TO_LABEL[self.dataset]
        names = read_csv_column(csv_path, "name")
        opensets = read_csv_column(csv_path, "openset")
        name2gt = {}
        for name, raw in zip(names, opensets):
            tokens = string_to_list(raw)
            name2gt[name] = "[" + ", ".join(tokens) + "]"
        return name2gt


class OVMERDPlusDataset(BaseEvalDataset):
    dataset = "OVMERDPlus"

    def get_test_name2gt(self):
        csv_path = self.cfg.PATH_TO_LABEL[self.dataset]
        df = pd.read_csv(csv_path)
        name2gt = {}
        for _, row in df.iterrows():
            name = row.get("name", "")
            if not name:
                continue
            tokens = string_to_list(row.get("openset", ""))
            if not tokens:
                tokens = ["neutral"]
            name2gt[name] = "[" + ", ".join(tokens) + "]"
        return name2gt


DATASET_BUILDERS = {
    "MER2023": MER2023Dataset,
    "MER2024": MER2024Dataset,
    "MELD": MELDDataset,
    "IEMOCAPFOUR": IEMOCAPFourDataset,
    "CMUMOSI": CMUMOSIDataset,
    "CMUMOSEI": CMUMOSEIDataset,
    "SIMS": SIMSDataset,
    "SIMSV2": SIMSv2Dataset,
    "MER2025OV": MER2025OVDataset,
    "OVMERDPLUS": OVMERDPlusDataset,
}


def build_dataset_map():
    return DATASET_BUILDERS.copy()

