import torch
import re
import numpy as np
import pandas as pd
import os
import glob
import sys
from types import SimpleNamespace

# --- Config Mocking for AffectGPT ---
# We need to mock the config module so that wheel_metrics can be imported and used.
if "config" not in sys.modules:
    cfg = SimpleNamespace()
    # Use absolute path to ensure it works from anywhere
    # Based on file location: HumanOmniV2/affect_r1/affect_reward.py
    # Wheel root: HumanOmniV2/affect_r1/emotion_wheel
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.EMOTION_WHEEL_ROOT = os.path.join(current_dir, "emotion_wheel")
    sys.modules["config"] = cfg

# Import wheel metrics from local merbench
# Ensure PYTHONPATH includes HumanOmniV2/affect_r1 so we can import merbench
try:
    from merbench.affectgpt_local.wheel_metrics import _map_label, _normalize_words
except ImportError:
    # If not in path, try adding current dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from merbench.affectgpt_local.wheel_metrics import _map_label, _normalize_words

# --- Reward Functions ---

def extract_answer(text):
    # Extract content between <answer> and </answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def calculate_f1(preds, gts):
    if not preds or not gts:
        return 0.0
    
    pred_set = set(preds)
    gt_set = set(gts)
    
    intersection = len(pred_set & gt_set)
    if intersection == 0:
        return 0.0
        
    precision = intersection / len(pred_set)
    recall = intersection / len(gt_set)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def emotion_wheel_reward(completions, **kwargs):
    """
    Reward function based on Emotion Wheel proximity.
    Uses AffectGPT's wheel mapping logic.
    
    Note: In GRPO, each prompt generates multiple completions, but ground_truths
    contains only one label per prompt. We need to repeat the ground truth for
    each generated completion.
    """
    rewards = []
    ground_truths = kwargs.get('openset', [])
    
    # Handle GRPO case: repeat ground truth for each completion
    # If we have N completions and M ground truths where N > M,
    # we assume each ground truth should be used for (N // M) completions
    if len(completions) > len(ground_truths) and len(ground_truths) > 0:
        repeat_factor = len(completions) // len(ground_truths)
        ground_truths = [gt for gt in ground_truths for _ in range(repeat_factor)]
    
    # Define the metrics we want to average over
    # AffectGPT usually uses case3_wheel{i}_level1 or level2
    # Here we use a robust average of wheel 1-5 level 1
    metrics = [f"case3_wheel{i}_level1" for i in range(1, 6)]
    
    for completion, gt_raw_list in zip(completions, ground_truths):
        content = completion[0]["content"]
        pred_text = extract_answer(content)
        
        # Parse prediction
        # Assume comma separated
        pred_raw_list = [p.strip() for p in pred_text.split(',')]
        
        # Calculate average F1 across all wheel metrics
        f1_scores = []
        
        for metric in metrics:
            # 1. Normalize and Map GT
            # gt_raw_list is already a list of strings from the dataset
            gt_norm = _normalize_words(gt_raw_list)
            gt_mapped = []
            for label in gt_norm:
                mapped = _map_label(label, metric)
                if mapped:
                    gt_mapped.append(mapped)
            
            # 2. Normalize and Map Pred
            pred_norm = _normalize_words(pred_raw_list)
            pred_mapped = []
            for label in pred_norm:
                mapped = _map_label(label, metric)
                if mapped:
                    pred_mapped.append(mapped)
            
            # 3. Calculate F1
            f1 = calculate_f1(pred_mapped, gt_mapped)
            f1_scores.append(f1)
            
        # Final reward is the mean of F1 scores across wheels
        rewards.append(float(np.mean(f1_scores)))
            
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Require <think>...</think> followed by <answer>...</answer>
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
