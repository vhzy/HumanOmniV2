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

# Import PAPO masking utilities
AFFECT_R1_ROOT = REPO_ROOT / "affect_r1"
if AFFECT_R1_ROOT.exists():
    sys.path.insert(0, str(AFFECT_R1_ROOT))

from papo_utils import mask_all_multimodal, mask_visual_inputs, mask_audio_inputs

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


def build_prompt_text(subtitle: str, early_answer: bool = False, 
                      mask_modality: str = "none", prompt_mode: str = "default") -> str:
    """
    Build prompt text with optional early answer mode and prompt customization.
    
    Args:
        subtitle: Subtitle text
        early_answer: Not used here - early answer is handled by adding assistant message
        mask_modality: Type of masking applied (none, all, visual, audio)
        prompt_mode: Prompt mode - "default" (original prompt) or "ask_masked" (ask about masked modality)
        
    Returns:
        Formatted prompt text
    """
    # Choose question based on prompt_mode and mask_modality

    if prompt_mode == "ask_masked" and mask_modality in ["visual", "audio"]:
        # Ask_masked mode: simple and direct, no subtitle, no type_template
        if mask_modality == "visual":
            # When visual is masked, explicitly ask what they can see
            question = (
                "Can you see any visual information in the video? Answer Yes or No."
            )
        elif mask_modality == "audio":
            # When audio is masked, explicitly ask what they can hear
            question = (
                "Can you hear any audio in the video? Answer Yes or No."
            )
        # For ask_masked mode: only return the question, no subtitle, no type_template
        return question
    else:
        # Default mode: original behavior with subtitle and type_template
        subtitle_prompt = ""
        if subtitle and subtitle.strip():
            subtitle_prompt = f"\nThe subtitle of this video is: <Subtitle>{subtitle.strip()}</Subtitle>."
        
        question = (
            "As an expert in the field of emotions, please focus on the facial expressions, body movements, tone, "
            "subtitle content, etc., in the video to discern clues related to the emotions of the individual. "
            "Please provide a detailed description and ultimately predict the emotional state of the individual in the video."
        )
        
        type_template = " Please provide the words to describe emotions within the  <answer> </answer> tags."
        
        # Build the prompt
        prompt = f"{subtitle_prompt}\n{question}\n{type_template}"
        
        return prompt


def build_messages(video_path: str, audio_path: str, subtitle: str, 
                   use_audio_in_video: bool = False, early_answer: bool = False,
                   mask_modality: str = "none", prompt_mode: str = "default"):
    """
    Build messages for model input with optional early answer mode and prompt customization.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        subtitle: Subtitle text
        use_audio_in_video: Whether to use audio from video
        early_answer: Whether to use early answer mode
        mask_modality: Type of masking applied (none, all, visual, audio)
        prompt_mode: Prompt mode - "default" or "ask_masked"
        
    Returns:
        List of message dictionaries
    """
    text_prompt = build_prompt_text(subtitle, early_answer=early_answer, 
                                    mask_modality=mask_modality, prompt_mode=prompt_mode)
    
    # Choose system prompt based on prompt_mode
    if prompt_mode == "ask_masked":
        # Simple system prompt for ask_masked mode
        system_prompt = "You are a helpful assistant."
    else:
        # Original affect system prompt for default mode
        system_prompt = AFFECT_SYSTEM_PROMPT
    
    # Prepare text prefix based on prompt_mode
    if prompt_mode == "ask_masked" :
        # Ask_masked mode: no prefix, just the question
        text_prefix = ""
    else:
        # Default mode: use descriptive prefix
        text_prefix = "Here is a video, with the audio from the video.\n"
    
    if use_audio_in_video:
        has_separate_audio = (audio_path is not None and audio_path != video_path)
        
        if has_separate_audio:
            # Case 1: Separate audio file
            content = [
                {"type": "video", "video": video_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": text_prefix + text_prompt}
            ]
        else:
            # Case 2: Audio from video
            video_audio_available = False
            if video_path:
                video_audio_available = check_if_video_has_audio(video_path)
            
            if video_audio_available:
                content = [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": video_path},
                    {"type": "text", "text": text_prefix + text_prompt}
                ]
            else:
                # No audio available
                if prompt_mode == "ask_masked" :
                    # Ask_masked mode: just use the prompt
                    content = [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": text_prompt}
                    ]
                else:
                    # Default mode: mention no audio
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

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    
    # Early answer mode: Add assistant message with <think> </think><answer> prefix
    # This will be placed after <|im_start|>assistant\n in the chat template
    if early_answer:
        messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": "<think> </think><answer>"}]
        })
    
    return messages


def prepare_inputs(processor, messages, use_audio_in_video):
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    
    # If last message is from assistant (early answer mode), don't add generation prompt
    # The assistant message already contains the prefix we want
    has_assistant_message = messages[-1]["role"] == "assistant"
    add_gen_prompt = not has_assistant_message
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen_prompt)
    # Ensure text is a string (apply_chat_template might return a list)
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    
    # For early answer mode: remove the trailing <|im_end|> tag
    # We want the model to continue generating from the prefilled <answer> tag
    if has_assistant_message:
        # Remove <|im_end|>\n from the end
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")]
        elif text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
    
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


def apply_counterfactual_mask(model_inputs: dict, mask_modality: str, 
                              mask_ratio: float, mask_noise: bool) -> dict:
    """
    Apply counterfactual masking to model inputs.
    
    Args:
        model_inputs: Dictionary of model inputs
        mask_modality: Type of masking ("none", "all", "visual", "audio")
        mask_ratio: Ratio of features to mask (0-1)
        mask_noise: Whether to use noise instead of zeros
        
    Returns:
        Dictionary with masked inputs
    """
    if mask_modality == "none":
        return model_inputs
    
    elif mask_modality == "all":
        return mask_all_multimodal(model_inputs, mask_ratio=mask_ratio, noise=mask_noise)
    
    elif mask_modality == "visual":
        return mask_visual_inputs(model_inputs, mask_ratio=mask_ratio, noise=mask_noise)
    
    elif mask_modality == "audio":
        return mask_audio_inputs(model_inputs, mask_ratio=mask_ratio, noise=mask_noise)
    
    else:
        raise ValueError(f"Unknown mask_modality: {mask_modality}")


def main():
    parser = argparse.ArgumentParser(description="Run MER-UniBench counterfactual inference with causal interventions.")
    
    # Original arguments
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
    
    # Counterfactual intervention arguments
    parser.add_argument("--mask-modality", 
                       choices=["none", "all", "visual", "audio"], 
                       default="none",
                       help="Type of modality masking: none (no mask), all (both), visual (video only), audio (audio only)")
    parser.add_argument("--mask-ratio", 
                       type=float, 
                       default=0.9,
                       help="Ratio of features to mask (default: 0.9 = 90%%)")
    parser.add_argument("--mask-noise", 
                       action="store_true", 
                       default=False,
                       help="Use noise instead of zeros for masking")
    parser.add_argument("--early-answer", 
                       action="store_true", 
                       default=False,
                       help="Use early answer mode (add empty <think></think> in prompt)")
    parser.add_argument("--prompt-mode",
                       choices=["default", "ask_masked"],
                       default="default",
                       help="Prompt mode: 'default' (original prompt) or 'ask_masked' (explicitly ask about masked modality)")
    
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
    
    print("=" * 80)
    print("Counterfactual Inference Configuration:")
    print(f"  Mask Modality: {args.mask_modality}")
    print(f"  Mask Ratio: {args.mask_ratio}")
    print(f"  Mask Noise: {args.mask_noise}")
    print(f"  Early Answer: {args.early_answer}")
    print("=" * 80)
    
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

        # Store counterfactual configuration in metadata
        extra_metadata = {
            "model_path": args.model_path,
            "counterfactual_config": {
                "mask_modality": args.mask_modality,
                "mask_ratio": args.mask_ratio,
                "mask_noise": args.mask_noise,
                "early_answer": args.early_answer,
                "prompt_mode": args.prompt_mode,
            }
        }

        writer = MerBenchResultWriter(
            output_root=args.output_root,
            dataset=dataset,
            run_name=args.run_name,
            checkpoint_name=args.checkpoint_name,
            extra_metadata=extra_metadata,
        )
        try:
            progress = tqdm(test_names, desc=f"{dataset} counterfactual inference")
            for name in progress:
                video_path = find_media(video_dir, name, [".mp4", ".avi", ".mkv"])
                audio_path = find_media(audio_dir, name, [".wav", ".mp3"])
                if video_path is None and audio_path is None:
                    progress.write(f"[WARN] Missing media for sample {name}, skipping.")
                    continue

                subtitle = subtitles.get(name, "")
                
                # Build messages with early_answer flag and prompt_mode
                messages = build_messages(
                    video_path, audio_path, subtitle, 
                    use_audio_in_video=args.use_audio_in_video,
                    early_answer=args.early_answer,
                    mask_modality=args.mask_modality,
                    prompt_mode=args.prompt_mode
                )
                
                has_separate_audio = (audio_path is not None)
                use_audio_in_video_for_processing = False if has_separate_audio else args.use_audio_in_video
                
                model_inputs = prepare_inputs(processor, messages, use_audio_in_video_for_processing)
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

                # Apply counterfactual masking
                model_inputs = apply_counterfactual_mask(
                    model_inputs, 
                    mask_modality=args.mask_modality,
                    mask_ratio=args.mask_ratio,
                    mask_noise=args.mask_noise
                )

                with torch.inference_mode():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_audio_in_video=use_audio_in_video_for_processing,  # 使用与输入处理一致的参数
                    )
                input_len = model_inputs["input_ids"].shape[-1]
                generated = outputs[0]
                response_ids = generated[input_len:] if generated.shape[0] > input_len else generated
                response = processor.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                # For early answer mode: prepend the prefilled content to make complete format
                # The model only generates the continuation, but we need the full response for parsing
                if args.early_answer:
                    response = "<think> </think><answer>" + response
                
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

