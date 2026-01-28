import torch
import torch.nn.functional as F
import re
import numpy as np
import pandas as pd
import os
import glob
import sys
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoModel, AutoTokenizer

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


# ------------------------- Rubric-based perceptual / coherence rewards -------------------------

# 默认路径与设备，可通过环境变量覆盖
_DEFAULT_EMBEDDER_PATH = "/mnt/afs/hanzhiyuan/huggingface/Qwen3-Embedding-0.6B"
_DEFAULT_RUBRIC_EMB_PATH = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_rubric_embeddings.pt"
_RUBRIC_DB = None
_EMBEDDER = None
_EMBEDDER_TOKENIZER = None
_TEXT_EMB_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_TEXT_EMB_CACHE_SIZE = 2048
_TASK_INSTRUCT = os.getenv("RUBRIC_INSTRUCT", "Given a clause query, calculate the similarity between this clause and the key clue.")
_DEBUG_MODE = os.getenv("RUBRIC_DEBUG", "0") == "1"

# Perception reward 计算中的相似度阈值，与 PAPO routing threshold 保持一致
# 可通过 set_score_threshold() 函数设置，默认值 0.5
_SCORE_THRESHOLD = 0.5

def set_score_threshold(threshold: float):
    """设置 perception reward 计算中的相似度阈值，与 PAPO routing threshold 保持一致。"""
    global _SCORE_THRESHOLD
    _SCORE_THRESHOLD = threshold
    print(f"[affect_reward] Score threshold set to {threshold}")


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对齐官方用法：取有效 token 的最后一位（左填充则直接取末位），再做池化。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_len]


def _get_device():
    env = os.getenv("RUBRIC_EMB_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_embedder():
    """惰性加载 Qwen3-Embedding."""
    global _EMBEDDER, _EMBEDDER_TOKENIZER
    if _EMBEDDER is not None:
        return

    model_path = _DEFAULT_EMBEDDER_PATH
    device = _get_device()
    _EMBEDDER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    
    # 关键修复：避免 DeepSpeed ZeRO-3 自动分片 embedder 模型
    # 策略：强制使用传统加载方式，禁用所有自动设备分配和内存优化
    # 使用 float32 保持与预加载的 rubric embeddings 一致
    _EMBEDDER = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,           # 禁用 accelerate 的自动设备分配
        low_cpu_mem_usage=False    # 禁用内存优化（避免触发 DeepSpeed 钩子）
    )
    _EMBEDDER = _EMBEDDER.to(device)
    _EMBEDDER.eval()
    
    # 额外保护：确保所有参数都是正确的形状
    for name, param in _EMBEDDER.named_parameters():
        # if param.dim() == 1 and 'weight' in name:
        #     print(f"Warning: Parameter {name} has unexpected shape {param.shape}")
        param.requires_grad = False  # 明确禁用梯度，避免 DeepSpeed 管理
    # 确保左侧 padding，与官方示例一致
    _EMBEDDER_TOKENIZER.padding_side = "left"


def _encode_texts(texts):
    """批量编码文本为单位向量张量。"""
    if len(texts) == 0:
        return None
    _load_embedder()
    device = _get_device()

    # 先查缓存，未命中再编码
    to_encode, order = [], []
    for t in texts:
        if t in _TEXT_EMB_CACHE:
            # LRU：移动到末尾
            _TEXT_EMB_CACHE.move_to_end(t)
        else:
            to_encode.append(t)
        order.append(t)

    new_embs = []
    if to_encode:
        with torch.inference_mode():
            inputs = _EMBEDDER_TOKENIZER(
                [f"Instruct: {_TASK_INSTRUCT}\nQuery: {t}" for t in to_encode],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            
            # 关键修复：在推理时确保参数完整（如果被 DeepSpeed ZeRO-3 分片）
            import sys
            if 'deepspeed' in sys.modules:
                import deepspeed
                # 使用 GatheredParameters 收集所有被分片的参数
                with deepspeed.zero.GatheredParameters(list(_EMBEDDER.parameters()), modifier_rank=None):
                    outputs = _EMBEDDER(**inputs)
            else:
                outputs = _EMBEDDER(**inputs)
            
            # official: last_token_pool
            emb = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            # 统一转换为 float32，确保与预加载的 rubric embeddings 类型一致
            emb = emb.float()
        # 写回缓存
        for t, e in zip(to_encode, emb):
            if len(_TEXT_EMB_CACHE) >= _TEXT_EMB_CACHE_SIZE:
                _TEXT_EMB_CACHE.popitem(last=False)
            _TEXT_EMB_CACHE[t] = e.detach().cpu()

    # 按原始顺序组合
    merged = [_TEXT_EMB_CACHE[t] for t in order]
    return torch.stack(merged, dim=0)


def _load_rubric_db():
    """惰性加载离线缓存的 Rubric embedding."""
    global _RUBRIC_DB
    if _RUBRIC_DB is not None:
        return _RUBRIC_DB
    rubric_path = os.getenv("RUBRIC_EMB_PATH", _DEFAULT_RUBRIC_EMB_PATH)
    if rubric_path and os.path.exists(rubric_path):
        try:
            _RUBRIC_DB = torch.load(rubric_path, map_location="cpu")
        except Exception:
            _RUBRIC_DB = {}
    else:
        _RUBRIC_DB = {}
    return _RUBRIC_DB


def _extract_think_text(text: str) -> str:
    """提取 <think>...<think> 内容；找不到则返回全文。"""
    if not isinstance(text, str):
        return ""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _split_sentences(text: str):
    """按句切分，过滤空句。"""
    if not text:
        return []
    # 同时覆盖中英文句号/问号/感叹号/逗号以及换行
    parts = re.split(r"[。！？!?\.，,]+\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _build_rubric_from_clues(clues: dict):
    """从 extracted_clues 文本动态编码得到 embedding 字典。"""
    visual = clues.get("visual_clues") or []
    audio = clues.get("audio_clues") or []
    logic = clues.get("reasoning_emotions") or []

    rubric = {
        "visual": _encode_texts(visual),
        "audio": _encode_texts(audio),
        "logic": _encode_texts(logic),
    }
    return rubric


def _get_rubric(path: str, extracted_clues: dict):
    """优先用离线缓存，其次用在线编码（并缓存在内存）。"""
    db = _load_rubric_db()
    if path and path in db:
        return db[path]
    if extracted_clues:
        rubric = _build_rubric_from_clues(extracted_clues)
        # 只在内存中缓存一次，避免重复编码
        if path:
            db[path] = rubric
        return rubric
    return {"visual": None, "audio": None, "logic": None}


def _column_max_scores(pred_embs: torch.Tensor, gt_embs: torch.Tensor, return_sims=False):
    """
    矩阵列最大相似度并做 ReLU 阈值缩放。
    
    阈值使用模块级变量 _SCORE_THRESHOLD（默认 0.5），可通过 set_score_threshold() 设置。
    缩放公式: scores = relu(max_sim - threshold) * scale
    其中 scale = 1 / (1 - threshold)，使得当 max_sim=1 时 score=1。
    
    Args:
        pred_embs: 预测句子的 embedding (Ns, D)
        gt_embs: GT 线索的 embedding (Ng, D)
        return_sims: 是否返回相似度矩阵（用于调试）
    
    Returns:
        始终返回 (scores, sims_matrix)，如果不需要 sims 则为 None
    """
    if pred_embs is None or gt_embs is None or gt_embs.numel() == 0:
        return None, None
    # 保证单位向量，同时对齐 device 和 dtype
    pred = F.normalize(pred_embs.to(device=gt_embs.device, dtype=gt_embs.dtype), p=2, dim=1)
    gt = F.normalize(gt_embs, p=2, dim=1)
    sims = pred @ gt.T  # (Ns, Ng)
    max_per_gt = sims.max(dim=0).values
    # 使用模块级阈值，缩放因子 = 1/(1-threshold) 使得 max_sim=1 时 score=1
    threshold = _SCORE_THRESHOLD
    scale = 1.0 / (1.0 - threshold) if threshold < 1.0 else 2.0
    scores = torch.relu(max_per_gt - threshold) * scale
    if return_sims:
        return scores, sims
    return scores, None


def _compute_sent_embs(completion_text: str):
    think_text = _extract_think_text(completion_text)
    sentences = _split_sentences(think_text)
    if not sentences:
        sentences = [think_text] if think_text else []
    return _encode_texts(sentences)


def rubric_perc_reward(completions, **kwargs):
    """
    显式感知奖励 R_perc：
    对 visual/audio 线索做列最大匹配，动态平均。
    """
    rewards = []
    paths = kwargs.get("path") or [None] * len(completions)
    clues_list = kwargs.get("extracted_clues") or [None] * len(completions)
    if not isinstance(paths, list):
        paths = [paths] * len(completions)
    if not isinstance(clues_list, list):
        clues_list = [clues_list] * len(completions)

    for idx, (completion, path, clues) in enumerate(zip(completions, paths, clues_list)):
        text = completion[0]["content"]
        pred_embs = _compute_sent_embs(text)
        rubric = _get_rubric(path, clues or {})

        # 调试模式：打印第一个样本的详细信息
        if _DEBUG_MODE and idx == 0:
            think_text = _extract_think_text(text)
            sentences = _split_sentences(think_text)
            if not sentences:
                sentences = [think_text] if think_text else []
            
            print("\n" + "="*80)
            print("[RUBRIC_DEBUG] rubric_perc_reward - Sample 0")
            print("="*80)
            print(f"Generated sentences ({len(sentences)}):")
            for i, sent in enumerate(sentences):
                print(f"  [{i}] {sent[:100]}..." if len(sent) > 100 else f"  [{i}] {sent}")
            
            # Visual clues
            visual_clues = (clues or {}).get("visual_clues", []) if clues else []
            print(f"\nVisual clues ({len(visual_clues)}):")
            for i, clue in enumerate(visual_clues):
                print(f"  [{i}] {clue}")
            
            # Audio clues
            audio_clues = (clues or {}).get("audio_clues", []) if clues else []
            print(f"\nAudio clues ({len(audio_clues)}):")
            for i, clue in enumerate(audio_clues):
                print(f"  [{i}] {clue}")

        s_v, sims_v = _column_max_scores(pred_embs, rubric.get("visual"), return_sims=_DEBUG_MODE and idx == 0)
        s_a, sims_a = _column_max_scores(pred_embs, rubric.get("audio"), return_sims=_DEBUG_MODE and idx == 0)

        if _DEBUG_MODE and idx == 0:
            if sims_v is not None:
                print(f"\nVisual similarity matrix shape: {sims_v.shape}")
                print(f"Visual similarity matrix (first 5x5):\n{sims_v[:5, :5].cpu().numpy()}")
                print(f"Visual column max: {sims_v.max(dim=0).values.cpu().numpy()}")
                print(f"Visual soft-threshold scores: {s_v.cpu().numpy()}")
            if sims_a is not None:
                print(f"\nAudio similarity matrix shape: {sims_a.shape}")
                print(f"Audio similarity matrix (first 5x5):\n{sims_a[:5, :5].cpu().numpy()}")
                print(f"Audio column max: {sims_a.max(dim=0).values.cpu().numpy()}")
                print(f"Audio soft-threshold scores: {s_a.cpu().numpy()}")

        Iv = 1 if s_v is not None else 0
        Ia = 1 if s_a is not None else 0
        denom = Iv + Ia
        if denom == 0:
            rewards.append(0.0)
            if _DEBUG_MODE and idx == 0:
                print(f"\nFinal R_perc: 0.0 (no valid clues)")
                print("="*80 + "\n")
            continue

        Sv = s_v.mean().item() if s_v is not None else 0.0
        Sa = s_a.mean().item() if s_a is not None else 0.0
        r = (Iv * Sv + Ia * Sa) / denom
        rewards.append(float(r))
        
        if _DEBUG_MODE and idx == 0:
            print(f"\nIv={Iv}, Ia={Ia}, Sv={Sv:.4f}, Sa={Sa:.4f}")
            print(f"Final R_perc: {r:.4f}")
            print("="*80 + "\n")

    return rewards


def rubric_coh_reward(completions, **kwargs):
    """
    显式连贯性奖励 R_coh：仅使用逻辑线索。
    """
    rewards = []
    paths = kwargs.get("path") or [None] * len(completions)
    clues_list = kwargs.get("extracted_clues") or [None] * len(completions)
    if not isinstance(paths, list):
        paths = [paths] * len(completions)
    if not isinstance(clues_list, list):
        clues_list = [clues_list] * len(completions)

    for idx, (completion, path, clues) in enumerate(zip(completions, paths, clues_list)):
        text = completion[0]["content"]
        pred_embs = _compute_sent_embs(text)
        rubric = _get_rubric(path, clues or {})

        # 调试模式：打印第一个样本的详细信息
        if _DEBUG_MODE and idx == 0:
            think_text = _extract_think_text(text)
            sentences = _split_sentences(think_text)
            if not sentences:
                sentences = [think_text] if think_text else []
            
            print("\n" + "="*80)
            print("[RUBRIC_DEBUG] rubric_coh_reward - Sample 0")
            print("="*80)
            print(f"Generated sentences ({len(sentences)}):")
            for i, sent in enumerate(sentences):
                print(f"  [{i}] {sent[:100]}..." if len(sent) > 100 else f"  [{i}] {sent}")
            
            # Logic/reasoning clues
            logic_clues = (clues or {}).get("reasoning_emotions", []) if clues else []
            print(f"\nReasoning emotion clues ({len(logic_clues)}):")
            for i, clue in enumerate(logic_clues):
                print(f"  [{i}] {clue}")

        s_l, sims_l = _column_max_scores(pred_embs, rubric.get("logic"), return_sims=_DEBUG_MODE and idx == 0)
        
        if _DEBUG_MODE and idx == 0:
            if sims_l is not None:
                print(f"\nLogic similarity matrix shape: {sims_l.shape}")
                print(f"Logic similarity matrix (first 5x5):\n{sims_l[:5, :5].cpu().numpy()}")
                print(f"Logic column max: {sims_l.max(dim=0).values.cpu().numpy()}")
                print(f"Logic soft-threshold scores: {s_l.cpu().numpy()}")
        
        if s_l is None:
            rewards.append(0.0)
            if _DEBUG_MODE and idx == 0:
                print(f"\nFinal R_coh: 0.0 (no valid clues)")
                print("="*80 + "\n")
        else:
            r = float(s_l.mean().item())
            rewards.append(r)
            if _DEBUG_MODE and idx == 0:
                print(f"\nFinal R_coh: {r:.4f}")
                print("="*80 + "\n")

    return rewards


# ========================== PAPO V2 Support: Similarity Matrices ==========================

def compute_similarity_matrices_for_papo(completions, **kwargs):
    """
    Compute similarity matrices for PAPO V2 routing.
    
    Returns:
        List of dicts, each containing:
        - 'sim_matrix_v': (Ns, Ng_v) visual similarity matrix
        - 'sim_matrix_a': (Ns, Ng_a) audio similarity matrix
        - 'sentences': list of sentence strings
        - 'score_v': (Ns,) max visual scores per sentence
        - 'score_a': (Ns,) max audio scores per sentence
    """
    results = []
    paths = kwargs.get("path") or [None] * len(completions)
    clues_list = kwargs.get("extracted_clues") or [None] * len(completions)
    
    if not isinstance(paths, list):
        paths = [paths] * len(completions)
    if not isinstance(clues_list, list):
        clues_list = [clues_list] * len(completions)
    
    for completion, path, clues in zip(completions, paths, clues_list):
        text = completion[0]["content"]
        
        # Extract think text and split into sentences
        think_text = _extract_think_text(text)
        sentences = _split_sentences(think_text)
        if not sentences:
            sentences = [think_text] if think_text else []
        
        # Get embeddings
        pred_embs = _encode_texts(sentences) if sentences else None
        rubric = _get_rubric(path, clues or {})
        
        result = {
            'sentences': sentences,
            'sim_matrix_v': None,
            'sim_matrix_a': None,
            'score_v': None,
            'score_a': None,
        }
        
        if pred_embs is not None:
            # Visual similarity matrix
            visual_embs = rubric.get("visual")
            if visual_embs is not None and visual_embs.numel() > 0:
                pred = F.normalize(pred_embs.to(device=visual_embs.device, dtype=visual_embs.dtype), p=2, dim=1)
                gt = F.normalize(visual_embs, p=2, dim=1)
                sims_v = pred @ gt.T  # (Ns, Ng_v)
                result['sim_matrix_v'] = sims_v
                result['score_v'] = sims_v.max(dim=-1).values
            
            # Audio similarity matrix
            audio_embs = rubric.get("audio")
            if audio_embs is not None and audio_embs.numel() > 0:
                pred = F.normalize(pred_embs.to(device=audio_embs.device, dtype=audio_embs.dtype), p=2, dim=1)
                gt = F.normalize(audio_embs, p=2, dim=1)
                sims_a = pred @ gt.T  # (Ns, Ng_a)
                result['sim_matrix_a'] = sims_a
                result['score_a'] = sims_a.max(dim=-1).values
        
        results.append(result)
    
    return results


def compute_modality_token_masks(
    sim_results: list,
    completion_ids: torch.Tensor,
    tokenizer,
    threshold: float = 0.5,
):
    """
    Compute token-level modality masks for PAPO V2.
    
    This function maps sentence-level modality assignments to token-level masks.
    Uses precise token counts from tokenizer for accurate mapping.
    
    Args:
        sim_results: List of similarity matrix results from compute_similarity_matrices_for_papo
        completion_ids: Token IDs of completions (B, L)
        tokenizer: Tokenizer for encoding sentences
        threshold: Similarity threshold for modality routing
        
    Returns:
        Dict with 'visual_token_mask' and 'audio_token_mask', both (B, L)
    """
    batch_size, seq_len = completion_ids.shape
    device = completion_ids.device
    
    visual_masks = []
    audio_masks = []
    
    for b in range(batch_size):
        if b >= len(sim_results):
            # No similarity data for this sample
            visual_masks.append(torch.zeros(seq_len, device=device))
            audio_masks.append(torch.zeros(seq_len, device=device))
            continue
        
        result = sim_results[b]
        sentences = result.get('sentences', [])
        score_v = result.get('score_v')
        score_a = result.get('score_a')
        
        if not sentences or (score_v is None and score_a is None):
            # No valid data, default to all tokens
            visual_masks.append(torch.ones(seq_len, device=device))
            audio_masks.append(torch.ones(seq_len, device=device))
            continue
        
        # Handle None scores and ensure correct device
        if score_v is None:
            score_v = torch.full((len(sentences),), float('-inf'), device=device)
        else:
            score_v = score_v.to(device)
        
        if score_a is None:
            score_a = torch.full((len(sentences),), float('-inf'), device=device)
        else:
            score_a = score_a.to(device)
        
        # Compute modality assignment per sentence
        is_visual = (score_v > threshold) & (score_v > score_a)
        is_audio = (score_a > threshold) & (score_a > score_v)
        
        # === Precise token count mapping ===
        # Use tokenizer to get exact token count for each sentence
        sentence_token_counts = []
        for sent in sentences:
            # Encode each sentence to get its token count
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            sentence_token_counts.append(len(tokens))
        
        total_sentence_tokens = sum(sentence_token_counts)
        
        # Initialize masks
        visual_mask = torch.zeros(seq_len, device=device)
        audio_mask = torch.zeros(seq_len, device=device)
        
        if total_sentence_tokens > 0:
            # Allocate completion tokens proportionally to each sentence's token count
            current_token = 0
            for i, token_count in enumerate(sentence_token_counts):
                if i >= len(is_visual):
                    break
                
                # Calculate token allocation proportionally
                # token_allocation = seq_len * (token_count / total_sentence_tokens)
                token_allocation = int(round(seq_len * token_count / total_sentence_tokens))
                token_start = current_token
                token_end = min(current_token + token_allocation, seq_len)
                
                # Ensure we don't exceed seq_len and handle last sentence
                if i == len(sentences) - 1:
                    token_end = seq_len  # Last sentence gets remaining tokens
                
                # Set masks for this sentence based on modality
                if token_start < token_end:
                    if is_visual[i]:
                        visual_mask[token_start:token_end] = 1.0
                    if is_audio[i]:
                        audio_mask[token_start:token_end] = 1.0
                
                current_token = token_end
        else:
            # Fallback: equal division if no tokens (shouldn't happen)
            tokens_per_sentence = seq_len // max(len(sentences), 1)
            for i, (vis, aud) in enumerate(zip(is_visual, is_audio)):
                start = i * tokens_per_sentence
                end = min((i + 1) * tokens_per_sentence, seq_len)
                if vis:
                    visual_mask[start:end] = 1.0
                if aud:
                    audio_mask[start:end] = 1.0
        
        visual_masks.append(visual_mask)
        audio_masks.append(audio_mask)
    
    return {
        'visual_token_mask': torch.stack(visual_masks, dim=0),
        'audio_token_mask': torch.stack(audio_masks, dim=0),
    }


def rubric_perc_reward_with_matrices(completions, **kwargs):
    """
    Extended version of rubric_perc_reward that also returns similarity matrices.
    
    Returns:
        Tuple of (rewards, sim_results)
    """
    rewards = rubric_perc_reward(completions, **kwargs)
    sim_results = compute_similarity_matrices_for_papo(completions, **kwargs)
    return rewards, sim_results
