"""Helpers for aligning AffectGPT config paths with local dataset layout."""

import os
from typing import Dict


DATASET_SUBDIRS = {
    # "MER2025OV": "mer2025/mer2025-dataset",
    # "MERCaptionPlus": "mer2025/mer2025-dataset",
    # "OVMERD": "mer2025/mer2025-dataset",
    "MER2023": "mer2025/mer2023-dataset-process",
    "MER2024": "mer2025/mer2024-dataset-process",
    "IEMOCAPFour": "mer2025/iemocap-process",
    "CMUMOSI": "mer2025/cmumosi-process",
    "CMUMOSEI": "mer2025/cmumosei-process",
    "SIMS": "mer2025/sims-process",
    "SIMSv2": "mer2025/simsv2-process",
    "MELD": "mer2025/meld-process",
    "OVMERDPlus": "mer2025/ovmerdplus-process",
}


def _join(base: str, *parts: str) -> str:
    return os.path.join(base, *parts)


def _apply_suffix_map(target: Dict[str, str], data_dir: Dict[str, str], suffix_map: Dict[str, str]):
    for key, suffix in suffix_map.items():
        if key in data_dir:
            target[key] = _join(data_dir[key], suffix)


def update_affectgpt_paths(config_module, dataset_root: str, use_face_video: bool = False):
    """Override AffectGPT config paths to match local dataset layout."""

    dataset_root = os.path.abspath(dataset_root)
    for name, subdir in DATASET_SUBDIRS.items():
        config_module.DATA_DIR[name] = _join(dataset_root, subdir)

    audio_suffix = {
        "MER2025OV": "audio",
        "MERCaptionPlus": "audio",
        "OVMERD": "audio",
        "MER2023": "audio",
        "MER2024": "audio",
        "IEMOCAPFour": "subaudio",
        "CMUMOSI": "subaudio",
        "CMUMOSEI": "subaudio",
        "SIMS": "audio",
        "MELD": "subaudio",
        "SIMSv2": "audio",
        "OVMERDPlus": "audio",
    }
    _apply_suffix_map(config_module.PATH_TO_RAW_AUDIO, config_module.DATA_DIR, audio_suffix)

    video_suffix = {
        "MER2025OV": "video",
        "MERCaptionPlus": "video",
        "OVMERD": "video",
        "MER2023": "video",
        "MER2024": "video",
        "IEMOCAPFour": "subvideo-tgt",
        "CMUMOSI": "subvideo",
        "CMUMOSEI": "subvideo_new",
        "SIMS": "video",
        "MELD": "subvideo",
        "SIMSv2": "video_new",
        "OVMERDPlus": "video",
    }
    
    # Dynamic check for video_face override if enabled
    for key, suffix in video_suffix.items():
        base_dir = config_module.DATA_DIR.get(key)
        
        # Default to raw video
        final_video_dir = _join(base_dir, suffix) if base_dir else ""
        
        if use_face_video and base_dir:
            # Check if a sibling/child 'video_face' directory exists
            face_video_dir = _join(base_dir, "video_face")
            if os.path.exists(face_video_dir):
                print(f"[INFO] Using processed face video directory for {key}: {face_video_dir}")
                final_video_dir = face_video_dir
            else:
                print(f"[WARN] --use-face-video requested but {face_video_dir} not found for {key}. Falling back to raw video.")
        
        if base_dir:
            config_module.PATH_TO_RAW_VIDEO[key] = final_video_dir

    # _apply_suffix_map(config_module.PATH_TO_RAW_VIDEO, config_module.DATA_DIR, video_suffix)

    face_suffix = {name: "openface_face" for name in [
        "MER2025OV",
        "MERCaptionPlus",
        "OVMERD",
        "MER2023",
        "MER2024",
        "IEMOCAPFour",
        "CMUMOSI",
        "CMUMOSEI",
        "SIMS",
        "MELD",
        "SIMSv2",
        "OVMERDPlus",
    ]}
    _apply_suffix_map(config_module.PATH_TO_RAW_FACE, config_module.DATA_DIR, face_suffix)

    transcription_suffix = {
        "MER2025OV": "subtitle_chieng.csv",
        "MERCaptionPlus": "subtitle_chieng.csv",
        "OVMERD": "subtitle_chieng.csv",
        "MER2023": "transcription-engchi-polish.csv",
        "MER2024": "transcription_merge.csv",
        "IEMOCAPFour": "transcription-engchi-polish.csv",
        "CMUMOSI": "transcription-engchi-polish.csv",
        "CMUMOSEI": "transcription-engchi-polish.csv",
        "SIMS": "transcription-engchi-polish.csv",
        "MELD": "transcription-engchi-polish.csv",
        "SIMSv2": "transcription-engchi-polish.csv",
        "OVMERDPlus": "subtitle_eng.csv",
    }
    _apply_suffix_map(config_module.PATH_TO_TRANSCRIPTIONS, config_module.DATA_DIR, transcription_suffix)

    label_suffix = {
        "MER2025OV": "track2_test.csv",
        "MER2023": "label-6way.npz",
        "MER2024": "label-6way.npz",
        "IEMOCAPFour": "label_4way.npz",
        "CMUMOSI": "label.npz",
        "CMUMOSEI": "label.npz",
        "SIMS": "label.npz",
        "MELD": "label.npz",
        "SIMSv2": "label.npz",
        "OVMERDPlus": "ovlabel.csv",
    }
    _apply_suffix_map(config_module.PATH_TO_LABEL, config_module.DATA_DIR, label_suffix)
