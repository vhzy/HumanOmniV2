import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
import av

from .affect_config import create_local_config
from .config_utils import update_affectgpt_paths
from .result_writer import MerBenchResultWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
OPEN_R1_SRC = REPO_ROOT / "src" / "open-r1-multimodal" / "src"
if OPEN_R1_SRC.exists():
    sys.path.insert(0, str(OPEN_R1_SRC))

try:
    from merbench.prompts import AFFECT_SYSTEM_PROMPT
except Exception:  # pragma: no cover
    AFFECT_SYSTEM_PROMPT = (
        "You are an expert affective-computing assistant. Reason in <think> tags "
        "and output open-vocabulary emotions in <answer> tags."
    )


def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False


DATASET_ALIASES = {name.upper(): name for name in [
    "MER2023",
    "MER2024",
    "MELD",
    "IEMOCAPFour",
    "CMUMOSI",
    "CMUMOSEI",
    "SIMS",
    "SIMSv2",
    "MER2025OV",
    "OVMERDPlus",
]}


def normalize_dataset_name(name: str) -> str:
    key = (name or "").strip().upper()
    return DATASET_ALIASES.get(key, (name or "").strip())


def import_affect_config(root: str):
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    import config  # type: ignore

    return config


def load_config(args):
    if args.affectgpt_root:
        try:
            cfg = import_affect_config(args.affectgpt_root)
            update_affectgpt_paths(cfg, args.dataset_root, use_face_video=args.use_face_video)
            return cfg
        except Exception as err:
            print(f"[WARN] Failed to import config from {args.affectgpt_root}: {err}")
    cfg = create_local_config(args.dataset_root)
    # Ensure paths are updated for local config too if needed, though create_local_config might need its own handling
    # or we call update_affectgpt_paths on it as well if it follows the same structure.
    # Assuming create_local_config returns a module-like object that update_affectgpt_paths can handle:
    update_affectgpt_paths(cfg, args.dataset_root, use_face_video=args.use_face_video)
    return cfg


def load_names_from_csv(csv_path: str, name_col: str = "name"):
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if name_col not in df.columns:
        name_col = "name" if "name" in df.columns else df.columns[0]
    return [str(v).strip() for v in df[name_col].tolist() if str(v).strip()]


def load_names_from_npz(npz_path: str, key_candidates):
    if not npz_path or not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for key in key_candidates:
        if key in data:
            corpus = data[key].tolist()
            return [name for name in corpus]
    raise ValueError(f"Keys {key_candidates} not found in {npz_path}")


def load_test_names(dataset: str, config_module):
    dataset = normalize_dataset_name(dataset)
    dataset_upper = dataset.upper()
    label_path = config_module.PATH_TO_LABEL.get(dataset, "")

    if dataset_upper == "MER2025OV":
        return load_names_from_csv(label_path, "name")
    if dataset_upper == "OVMERDPLUS":
        subtitle_csv = config_module.PATH_TO_TRANSCRIPTIONS.get(dataset, "")
        return load_names_from_csv(subtitle_csv, "name")
    if dataset_upper in {"MER2023", "MER2024"}:
        return load_names_from_npz(label_path, ["test1_corpus"])
    if dataset_upper in {"MELD", "CMUMOSI", "CMUMOSEI", "SIMS", "SIMSV2"}:
        return load_names_from_npz(label_path, ["test_corpus"])
    if dataset_upper == "IEMOCAPFOUR":
        data = np.load(label_path, allow_pickle=True)["whole_corpus"].tolist()
        return [name for name in data if len(name) > 4 and name[4] == "5"]
    raise ValueError(f"Unsupported dataset {dataset}")


def read_subtitles(csv_path: str):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    name_col = "name" if "name" in df.columns else df.columns[0]
    text_col_candidates = ["english", "subtitle", "text", "Transcript"]
    text_col = next((c for c in text_col_candidates if c in df.columns), df.columns[-1])
    subtitles = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name:
            continue
        subtitle = row[text_col]
        subtitles[name] = "" if pd.isna(subtitle) else str(subtitle)
    return subtitles


def find_media(base_dir: str, name: str, extensions):
    if not base_dir:
        return None
    for ext in extensions:
        candidate = os.path.join(base_dir, f"{name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def build_prompt_text(subtitle: str) -> str:
    # Matches _make_conversation_image_and_video logic in grpo_qwenomni.py
    subtitle_prompt = ""
    if subtitle and subtitle.strip():
        subtitle_prompt = f"\nThe subtitle of this video is: <Subtitle>{subtitle.strip()}</Subtitle>."
    
    question = (
        "As an expert in the field of emotions, please focus on the facial expressions, body movements, tone, "
        "subtitle content, etc., in the video to discern clues related to the emotions of the individual. "
        "Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
    )
    
    type_template = " Please provide the words to describe emotions within the  <answer> </answer> tags."
    
    # In training: text_prompt = f"{subtitle_prompt}\n{question}\n" + self.TYPE_TEMPLATE[...]
    return f"{subtitle_prompt}\n{question}\n{type_template}"


def build_messages(video_path: str, audio_path: str, subtitle: str, use_audio_in_video: bool = False):
    text_prompt = build_prompt_text(subtitle)
    
    # Matches logic in grpo_qwenomni.py lines 285-370
    if use_audio_in_video:
        has_separate_audio = (audio_path is not None and audio_path != video_path)
        
        if has_separate_audio:
            # Case 1: Separate audio file
            content = [
                {"type": "video", "video": video_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": f"Here is a video, with the audio from the video.\n" + text_prompt}
            ]
        else:
            # Case 2: Audio from video
            video_audio_available = False
            if video_path:
                video_audio_available = check_if_video_has_audio(video_path)
            
            if video_audio_available:
                content = [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": video_path}, # Training uses video path as audio source here
                    {"type": "text", "text": f"Here is a video, with the audio from the video.\n" + text_prompt}
                ]
            else:
                content = [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": f"Here is the video, and there is no audio information, you don't need to process the audio.\n" + text_prompt}
                ]
    else:
        # Case 3: No audio used
        content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": text_prompt}
        ]

    return [
        {"role": "system", "content": [{"type": "text", "text": AFFECT_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def prepare_inputs(processor, messages, use_audio_in_video):
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Ensure text is a string (apply_chat_template might return a list)
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    inputs = processor(
        text=[text],
        images=images,
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Run MER-UniBench inference and store JSONL results.")
    parser.add_argument("--model-path", required=True, help="Path to finetuned Qwen2.5-Omni checkpoint.")
    parser.add_argument("--processor-path", default=None, help="Optional processor path (defaults to model).")
    parser.add_argument("--affectgpt-root", default=None, help="Optional path to AffectGPT/AffectGPT directory.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing MER datasets.")
    parser.add_argument("--output-root", default="output", help="Base directory to store inference outputs.")
    parser.add_argument("--run-name", required=True, help="Name of the current experiment/run.")
    parser.add_argument("--checkpoint-name", required=True, help="Logical checkpoint identifier, e.g., checkpoint_000001.")
    parser.add_argument("--datasets", default="MER2023,MER2024,MELD,IEMOCAPFour,CMUMOSI,CMUMOSEI,SIMS,SIMSV2,MER2025OV,OVMERDPlus")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--use-audio-in-video", action="store_true", default=False)
    parser.add_argument("--use-face-video", action="store_true", default=False, help="Use processed face videos (video_face) if available.")
    args = parser.parse_args()

    processor_path = args.processor_path or args.model_path
    # Try to use flash_attention_2 if available, otherwise fall back to default
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"  # Use SDPA as fallback
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)

    config_module = load_config(args)

    datasets = [normalize_dataset_name(item) for item in args.datasets.split(",") if item.strip()]
    for dataset in datasets:
        try:
            test_names = load_test_names(dataset, config_module)
        except Exception as err:
            print(f"[WARN] Failed to load names for {dataset}: {err}")
            continue
        subtitles = read_subtitles(config_module.PATH_TO_TRANSCRIPTIONS.get(dataset, ""))
        video_dir = config_module.PATH_TO_RAW_VIDEO.get(dataset, "")
        audio_dir = config_module.PATH_TO_RAW_AUDIO.get(dataset, "")

        writer = MerBenchResultWriter(
            output_root=args.output_root,
            dataset=dataset,
            run_name=args.run_name,
            checkpoint_name=args.checkpoint_name,
            extra_metadata={"model_path": args.model_path},
        )
        try:
            progress = tqdm(test_names, desc=f"{dataset} inference")
            for name in progress:
                video_path = find_media(video_dir, name, [".mp4", ".avi", ".mkv"])
                audio_path = find_media(audio_dir, name, [".wav", ".mp3"])
                if video_path is None and audio_path is None:
                    progress.write(f"[WARN] Missing media for sample {name}, skipping.")
                    continue

                subtitle = subtitles.get(name, "")
                
                # Logic update:
                # If use_audio_in_video is True:
                #   We don't need to force audio_path = video_path here manually because build_messages handles it
                #   based on the flag and video_path presence.
                # However, for "separate audio" detection, we need to know if the found audio_path 
                # is actually a separate file or just us falling back.
                
                # In find_media, audio_path is found from audio_dir. 
                # If audio_dir is different or file is different, it is separate.
                # If find_media returns None, and we used to set it to video_path...
                
                # Let's KEEP audio_path as the "found separate audio file" (or None).
                # And pass the flag to build_messages.
                
                # But wait, original code did:
                # if audio_path is None and args.use_audio_in_video and video_path: audio_path = video_path
                # We should REMOVE this overwrite so build_messages knows it's NOT a separate file.
                
                messages = build_messages(video_path, audio_path, subtitle, use_audio_in_video=args.use_audio_in_video)
                
                # We also need to correct the 'use_audio_in_video' flag passed to prepare_inputs/process_mm_info
                # to match training logic: False if separate audio exists, else True (if global flag is True).
                
                has_separate_audio = (audio_path is not None)
                use_audio_in_video_for_processing = False if has_separate_audio else args.use_audio_in_video
                
                model_inputs = prepare_inputs(processor, messages, use_audio_in_video_for_processing)
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_audio_in_video=args.use_audio_in_video, # This usually stays as the model config/arg
                    )
                input_len = model_inputs["input_ids"].shape[-1]
                generated = outputs[0]
                response_ids = generated[input_len:] if generated.shape[0] > input_len else generated
                response = processor.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                writer.log_sample(
                    name=name,
                    response_text=response,
                    subtitle=subtitle,
                    metadata={"video_path": video_path or "", "audio_path": audio_path or ""},
                )
        finally:
            writer.close()


if __name__ == "__main__":
    main()

