"""Local AffectGPT-style config for running inference without the upstream repo."""

import os
from pathlib import Path
from types import SimpleNamespace

from .config_utils import DATASET_SUBDIRS, update_affectgpt_paths


DEFAULT_LLM_PATHS = {
    "Qwen25": "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-7B-Instruct",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMOTION_WHEEL_ROOT = REPO_ROOT / "affect_r1" / "emotion_wheel"


def _init_dict():
    return {name: "" for name in DATASET_SUBDIRS}


def create_local_config(dataset_root: str, llm_paths=None):
    cfg = SimpleNamespace()
    cfg.DATA_DIR = _init_dict()
    cfg.PATH_TO_RAW_AUDIO = _init_dict()
    cfg.PATH_TO_RAW_VIDEO = _init_dict()
    cfg.PATH_TO_RAW_FACE = _init_dict()
    cfg.PATH_TO_TRANSCRIPTIONS = _init_dict()
    cfg.PATH_TO_LABEL = _init_dict()
    cfg.PATH_TO_LLM = llm_paths or DEFAULT_LLM_PATHS.copy()
    cfg.EMOTION_WHEEL_ROOT = os.path.abspath(os.fspath(DEFAULT_EMOTION_WHEEL_ROOT))

    update_affectgpt_paths(cfg, dataset_root)
    return cfg

