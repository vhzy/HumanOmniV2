# Copyright 2025 - PAPO for HumanOmniV2
# Perception-Aware Policy Optimization utilities for multimodal (audio + video) models
#
# Supports three versions:
#   V0: Mask both audio and video simultaneously
#   V1: Mask audio and video separately (dual-stream)
#   V2: Matrix-guided fine-grained PAPO (uses similarity matrices from Stage 2)

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
import re


class PAPOVersion(str, Enum):
    """PAPO version enum"""
    V0 = "v0"  # Mask both audio and video simultaneously
    V1 = "v1"  # Mask audio and video separately
    V2 = "v2"  # Matrix-guided fine-grained PAPO


# ========================== Multimodal Input Keys ==========================
# Qwen-Omni specific keys - using keyword matching approach
# This approach is more robust as it handles various key naming conventions

# Keywords for features that SHOULD be masked (actual feature tensors)
VISUAL_MASK_KEYWORDS = ["pixel", "image"]  # pixel_values, pixel_values_videos, image_features
AUDIO_MASK_KEYWORDS = ["audio", "input_features"]  # input_features (audio mel features)

# Keywords for metadata that should NOT be masked (must stay consistent with features)
SKIP_KEYWORDS = ["attention_mask", "lengths", "position", "grid", "second_per_grid"]

# Control flags that should never be modified
CONTROL_KEYS = ["use_audio_in_video", "rope_deltas"]


# ========================== Masking Functions ==========================

def _mask_tensor(tensor: torch.Tensor, mask_ratio: float = 0.9, noise: bool = False) -> torch.Tensor:
    """
    Mask a tensor by setting a portion of its values to zero or noise.
    
    Args:
        tensor: Input tensor to mask
        mask_ratio: Ratio of elements to mask (default 0.9 = 90%)
        noise: If True, replace with random noise instead of zeros
        
    Returns:
        Masked tensor
    """
    if tensor is None:
        return None
    
    # Create random mask
    mask = torch.rand_like(tensor.float()) > mask_ratio
    
    if noise:
        # Replace with random noise
        noise_tensor = torch.randn_like(tensor.float()) * 0.1
        return torch.where(mask, tensor.float(), noise_tensor).to(tensor.dtype)
    else:
        # Replace with zeros
        return tensor * mask.to(tensor.dtype)


def _should_mask_key(key: str, mask_keywords: List[str]) -> bool:
    """Check if a key should be masked based on keywords."""
    k_lower = key.lower()
    return any(kw in k_lower for kw in mask_keywords)


def _should_skip_key(key: str) -> bool:
    """Check if a key is metadata that should NOT be masked."""
    k_lower = key.lower()
    return any(kw in k_lower for kw in SKIP_KEYWORDS) or key in CONTROL_KEYS


def mask_all_multimodal(
    multimodal_inputs: Dict[str, Any],
    mask_ratio: float = 0.9,
    noise: bool = False
) -> Dict[str, Any]:
    """
    V0: Mask both audio and video features simultaneously.
    
    Strategy: Randomly mask mask_ratio portion of feature values, keeping metadata intact.
    - Avoids completely zeroing out which can cause audio/video processing to crash
    - Significantly weakens the influence of multimodal information
    
    Args:
        multimodal_inputs: Dictionary of multimodal inputs
        mask_ratio: Ratio of features to mask (default 0.9 = keep 10%)
        noise: If True, use noise instead of zeros
        
    Returns:
        Dictionary with masked multimodal feature inputs
    """
    masked = {}
    all_mask_keywords = VISUAL_MASK_KEYWORDS + AUDIO_MASK_KEYWORDS
    
    for key, value in multimodal_inputs.items():
        # Skip metadata keys (grid, lengths, attention_mask, etc.)
        if _should_skip_key(key):
            masked[key] = value
            continue
        
        # Check if this is a feature tensor that should be masked
        should_mask = _should_mask_key(key, all_mask_keywords)
        
        if should_mask and isinstance(value, torch.Tensor) and value.numel() > 0:
            masked[key] = _mask_tensor(value, mask_ratio, noise)
        else:
            masked[key] = value
            
    return masked


def mask_visual_inputs(
    multimodal_inputs: Dict[str, Any],
    mask_ratio: float = 0.9,
    noise: bool = False
) -> Dict[str, Any]:
    """
    V1: Mask only visual features (keep audio intact).
    
    Strategy: Randomly mask mask_ratio portion of visual feature values.
    Metadata (grid_thw, etc.) is preserved to maintain consistency.
    
    Args:
        multimodal_inputs: Dictionary of multimodal inputs
        mask_ratio: Ratio of visual features to mask (default 0.9 = keep 10%)
        noise: If True, use noise instead of zeros
        
    Returns:
        Dictionary with masked visual inputs (audio preserved)
    """
    masked = {}
    
    for key, value in multimodal_inputs.items():
        # Skip metadata keys
        if _should_skip_key(key):
            masked[key] = value
            continue
        
        # Check if this is a visual feature tensor
        should_mask = _should_mask_key(key, VISUAL_MASK_KEYWORDS)
        
        if should_mask and isinstance(value, torch.Tensor) and value.numel() > 0:
            masked[key] = _mask_tensor(value, mask_ratio, noise)
        else:
            # Keep audio and other features intact
            masked[key] = value
            
    return masked


def mask_audio_inputs(
    multimodal_inputs: Dict[str, Any],
    mask_ratio: float = 0.9,
    noise: bool = False
) -> Dict[str, Any]:
    """
    V1: Mask only audio features (keep visual intact).
    
    Strategy: Randomly mask mask_ratio portion of audio feature values.
    Metadata (feature_attention_mask, lengths) is preserved to maintain consistency.
    
    Args:
        multimodal_inputs: Dictionary of multimodal inputs
        mask_ratio: Ratio of audio features to mask (default 0.9 = keep 10%)
        noise: If True, use noise instead of zeros
        
    Returns:
        Dictionary with masked audio inputs (visual preserved)
    """
    masked = {}
    
    for key, value in multimodal_inputs.items():
        # Skip metadata keys
        if _should_skip_key(key):
            masked[key] = value
            continue
        
        # Check if this is an audio feature tensor
        should_mask = _should_mask_key(key, AUDIO_MASK_KEYWORDS)
        
        if should_mask and isinstance(value, torch.Tensor) and value.numel() > 0:
            masked[key] = _mask_tensor(value, mask_ratio, noise)
        else:
            # Keep visual and other features intact
            masked[key] = value
            
    return masked


# ========================== KL and Entropy Computation ==========================

def compute_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_masked: torch.Tensor,
    kl_penalty: str = "kl"
) -> torch.Tensor:
    """
    Compute KL divergence between full and masked log probabilities.
    
    Args:
        log_probs: Log probabilities from full input (B, L)
        log_probs_masked: Log probabilities from masked input (B, L)
        kl_penalty: Type of KL penalty ("kl", "abs", "mse", "low_var_kl")
        
    Returns:
        KL divergence tensor (B, L)
    """
    log_probs = log_probs.float()
    log_probs_masked = log_probs_masked.float()
    
    if kl_penalty == "kl":
        # Standard KL: log(p) - log(q)
        return log_probs - log_probs_masked
    
    elif kl_penalty == "abs":
        return (log_probs - log_probs_masked).abs()
    
    elif kl_penalty == "mse":
        return 0.5 * (log_probs - log_probs_masked).square()
    
    elif kl_penalty == "low_var_kl":
        # Low variance KL approximation
        kl = (log_probs_masked - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)
    
    else:
        raise ValueError(f"Unknown KL penalty type: {kl_penalty}")


def compute_entropy_loss(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy loss (negative mean log probability).
    
    Args:
        log_probs: Log probabilities (B, L)
        mask: Attention mask for valid tokens (B, L)
        
    Returns:
        Entropy loss scalar
    """
    # Entropy estimator: -mean(log_probs)
    masked_log_probs = log_probs * mask
    total_tokens = mask.sum()
    if total_tokens > 0:
        return -masked_log_probs.sum() / total_tokens
    return torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype)


def compute_papo_loss_v0(
    log_probs_full: torch.Tensor,
    log_probs_masked: torch.Tensor,
    completion_mask: torch.Tensor,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.03,
    kl_penalty: str = "kl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PAPO loss for V0 (both modalities masked together).
    
    Loss = -kl_coef * KL(full || masked) + entropy_coef * (H_full + H_masked)
    
    Args:
        log_probs_full: Log probs from full input (B, L)
        log_probs_masked: Log probs from masked input (B, L)
        completion_mask: Mask for completion tokens (B, L)
        kl_coef: Coefficient for KL term (maximize KL)
        entropy_coef: Coefficient for entropy regularization
        kl_penalty: Type of KL penalty
        
    Returns:
        (loss, metrics_dict)
    """
    # Compute KL divergence
    kl = compute_kl_divergence(log_probs_full, log_probs_masked, kl_penalty)
    kl_loss = (kl * completion_mask).sum() / completion_mask.sum().clamp(min=1)
    
    # Compute entropy losses
    entropy_full = compute_entropy_loss(log_probs_full, completion_mask)
    entropy_masked = compute_entropy_loss(log_probs_masked, completion_mask)
    entropy_total = entropy_full + entropy_masked
    
    # Compute individual loss components
    kl_term = -kl_coef * kl_loss  # This is what we subtract from loss (maximize KL)
    entropy_term = entropy_coef * entropy_total  # Entropy regularization
    
    # Final PAPO loss: maximize KL (negative sign) + entropy regularization
    loss = kl_term + entropy_term
    
    metrics = {
        # KL loss (higher = model relies more on multimodal info)
        "papo/kl_loss": kl_loss.detach().item(),
        # Entropy losses
        "papo/entropy_full": entropy_full.detach().item(),
        "papo/entropy_masked": entropy_masked.detach().item(),
        "papo/entropy_total": entropy_total.detach().item(),
        # Loss components
        "papo/kl_term": kl_term.detach().item(),
        "papo/entropy_term": entropy_term.detach().item(),
        "papo/loss": loss.detach().item(),
    }
    
    return loss, metrics


def compute_papo_loss_v1(
    log_probs_full: torch.Tensor,
    log_probs_no_v: torch.Tensor,
    log_probs_no_a: torch.Tensor,
    completion_mask: torch.Tensor,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.03,
    kl_penalty: str = "kl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PAPO loss for V1 (separate visual and audio masking).
    
    Loss = -kl_coef * (KL_v + KL_a) + entropy_coef * (H_full + H_no_v + H_no_a)
    
    Args:
        log_probs_full: Log probs from full input (B, L)
        log_probs_no_v: Log probs with visual masked (B, L)
        log_probs_no_a: Log probs with audio masked (B, L)
        completion_mask: Mask for completion tokens (B, L)
        kl_coef: Coefficient for KL terms
        entropy_coef: Coefficient for entropy regularization
        kl_penalty: Type of KL penalty
        
    Returns:
        (loss, metrics_dict)
    """
    # Compute KL divergences
    kl_v = compute_kl_divergence(log_probs_full, log_probs_no_v, kl_penalty)
    kl_a = compute_kl_divergence(log_probs_full, log_probs_no_a, kl_penalty)
    
    kl_v_loss = (kl_v * completion_mask).sum() / completion_mask.sum().clamp(min=1)
    kl_a_loss = (kl_a * completion_mask).sum() / completion_mask.sum().clamp(min=1)
    
    # Compute entropy losses
    entropy_full = compute_entropy_loss(log_probs_full, completion_mask)
    entropy_no_v = compute_entropy_loss(log_probs_no_v, completion_mask)
    entropy_no_a = compute_entropy_loss(log_probs_no_a, completion_mask)
    entropy_total = entropy_full + entropy_no_v + entropy_no_a
    
    # Compute individual loss components
    kl_total = kl_v_loss + kl_a_loss
    kl_term = -kl_coef * kl_total  # This is what we subtract from loss (maximize KL)
    entropy_term = entropy_coef * entropy_total  # Entropy regularization
    
    # Final PAPO loss
    loss = kl_term + entropy_term
    
    metrics = {
        # KL losses (higher = model relies more on multimodal info)
        "papo/kl_v_loss": kl_v_loss.detach().item(),
        "papo/kl_a_loss": kl_a_loss.detach().item(),
        "papo/kl_total": kl_total.detach().item(),
        # Entropy losses
        "papo/entropy_full": entropy_full.detach().item(),
        "papo/entropy_no_v": entropy_no_v.detach().item(),
        "papo/entropy_no_a": entropy_no_a.detach().item(),
        "papo/entropy_total": entropy_total.detach().item(),
        # Loss components
        "papo/kl_term": kl_term.detach().item(),
        "papo/entropy_term": entropy_term.detach().item(),
        "papo/loss": loss.detach().item(),
    }
    
    return loss, metrics


# ========================== V2: Matrix-Guided Fine-Grained PAPO ==========================

def extract_think_text(text: str) -> str:
    """Extract content from <think> tags."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    # Split on Chinese/English punctuation and newlines
    parts = re.split(r"[。！？!?\.，,]+\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_modality_routing_mask(
    sim_matrix_v: Optional[torch.Tensor],
    sim_matrix_a: Optional[torch.Tensor],
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute modality routing mask based on similarity matrices.
    
    For each sentence t:
    - If score_v > threshold AND score_v > score_a: Visual
    - If score_a > threshold AND score_a > score_v: Audio
    - Otherwise: Neutral (no KL loss)
    
    Args:
        sim_matrix_v: Visual similarity matrix (Ns, Ng_v) or None
        sim_matrix_a: Audio similarity matrix (Ns, Ng_a) or None
        threshold: Minimum similarity threshold
        
    Returns:
        (visual_mask, audio_mask) - both (Ns,) boolean tensors
    """
    # Handle empty/None cases
    if sim_matrix_v is not None and sim_matrix_v.numel() > 0:
        score_v = sim_matrix_v.max(dim=-1).values  # (Ns,)
    else:
        # No visual clues - set to -inf so audio can be selected
        if sim_matrix_a is not None:
            score_v = torch.full((sim_matrix_a.size(0),), float('-inf'), 
                                device=sim_matrix_a.device)
        else:
            return None, None
    
    if sim_matrix_a is not None and sim_matrix_a.numel() > 0:
        score_a = sim_matrix_a.max(dim=-1).values  # (Ns,)
    else:
        # No audio clues - set to -inf so visual can be selected
        score_a = torch.full_like(score_v, float('-inf'))
    
    # Classification logic
    is_visual = (score_v > threshold) & (score_v > score_a)
    is_audio = (score_a > threshold) & (score_a > score_v)
    
    return is_visual, is_audio


def sentence_mask_to_token_mask(
    sentence_mask: torch.Tensor,
    sentence_spans: List[Tuple[int, int]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert sentence-level mask to token-level mask.
    
    Args:
        sentence_mask: Boolean mask for sentences (Ns,)
        sentence_spans: List of (start, end) token positions for each sentence
        seq_len: Total sequence length
        device: Device for output tensor
        
    Returns:
        Token-level mask (seq_len,)
    """
    token_mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
    
    for i, (start, end) in enumerate(sentence_spans):
        if i < len(sentence_mask) and sentence_mask[i]:
            token_mask[start:end] = 1.0
            
    return token_mask


def compute_papo_loss_v2(
    log_probs_full: torch.Tensor,
    log_probs_no_v: torch.Tensor,
    log_probs_no_a: torch.Tensor,
    completion_mask: torch.Tensor,
    visual_token_mask: torch.Tensor,
    audio_token_mask: torch.Tensor,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.03,
    kl_penalty: str = "kl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PAPO loss for V2 (matrix-guided fine-grained).
    
    Only compute KL for tokens that are routed to the corresponding modality:
    - KL_v only for Visual tokens
    - KL_a only for Audio tokens
    - Neutral tokens don't contribute KL loss
    
    Entropy regularization is global (all tokens).
    
    Args:
        log_probs_full: Log probs from full input (B, L)
        log_probs_no_v: Log probs with visual masked (B, L)
        log_probs_no_a: Log probs with audio masked (B, L)
        completion_mask: Mask for completion tokens (B, L)
        visual_token_mask: Mask for visual-routed tokens (B, L)
        audio_token_mask: Mask for audio-routed tokens (B, L)
        kl_coef: Coefficient for KL terms
        entropy_coef: Coefficient for entropy regularization
        kl_penalty: Type of KL penalty
        
    Returns:
        (loss, metrics_dict)
    """
    # Compute KL divergences
    kl_v = compute_kl_divergence(log_probs_full, log_probs_no_v, kl_penalty)
    kl_a = compute_kl_divergence(log_probs_full, log_probs_no_a, kl_penalty)
    
    # Apply selective masking
    visual_mask_combined = completion_mask * visual_token_mask
    audio_mask_combined = completion_mask * audio_token_mask
    
    # Compute masked KL losses
    visual_tokens = visual_mask_combined.sum().clamp(min=1)
    audio_tokens = audio_mask_combined.sum().clamp(min=1)
    
    kl_v_loss = (kl_v * visual_mask_combined).sum() / visual_tokens
    kl_a_loss = (kl_a * audio_mask_combined).sum() / audio_tokens
    
    # Global entropy regularization (all tokens)
    entropy_full = compute_entropy_loss(log_probs_full, completion_mask)
    entropy_no_v = compute_entropy_loss(log_probs_no_v, completion_mask)
    entropy_no_a = compute_entropy_loss(log_probs_no_a, completion_mask)
    entropy_total = entropy_full + entropy_no_v + entropy_no_a
    
    # Compute individual loss components
    kl_total = kl_v_loss + kl_a_loss
    kl_term = -kl_coef * kl_total  # This is what we subtract from loss (maximize KL)
    entropy_term = entropy_coef * entropy_total  # Entropy regularization
    
    # Final PAPO loss
    loss = kl_term + entropy_term
    
    metrics = {
        # KL losses (higher = model relies more on multimodal info, which is good)
        "papo/kl_v_loss": kl_v_loss.detach().item(),
        "papo/kl_a_loss": kl_a_loss.detach().item(),
        "papo/kl_total": kl_total.detach().item(),
        # Token routing stats
        "papo/visual_tokens": visual_tokens.detach().item(),
        "papo/audio_tokens": audio_tokens.detach().item(),
        "papo/neutral_ratio": (1 - (visual_mask_combined + audio_mask_combined).sum() / completion_mask.sum().clamp(min=1)).detach().item(),
        # Entropy losses (for exploration)
        "papo/entropy_full": entropy_full.detach().item(),
        "papo/entropy_no_v": entropy_no_v.detach().item(),
        "papo/entropy_no_a": entropy_no_a.detach().item(),
        "papo/entropy_total": entropy_total.detach().item(),
        # Loss components
        "papo/kl_term": kl_term.detach().item(),  # -kl_coef * kl_total
        "papo/entropy_term": entropy_term.detach().item(),  # entropy_coef * entropy_total
        "papo/loss": loss.detach().item(),  # Total PAPO loss added to main loss
    }
    
    return loss, metrics


# ========================== Configuration ==========================

class PAPOConfig:
    """Configuration for PAPO algorithm
    
    Default values are aligned with original PAPO paper:
    - kl_coef: 0.001 (1e-3) - KL coefficient for perception loss
    - entropy_coef: 0.03 - Double entropy regularization coefficient
    - mask_ratio: 0.6 - Original PAPO uses 60% blackening probability
    - use_noise: False - Original PAPO sets patches to zeros (black)
    """
    
    def __init__(
        self,
        enabled: bool = False,
        version: str = "v1",
        mask_ratio: float = 0.9,       # Original PAPO: black_prob=0.6
        use_noise: bool = False,        # Original PAPO: sets to 0 (black), not noise
        kl_coef: float = 0.001,         # Original PAPO: kl_prcp_coef = 1e-3
        entropy_coef: float = 0.03,     # Original PAPO: aug_entropy_loss_coef = 0.03
        kl_penalty: str = "kl",
        # V2 specific
        routing_threshold: float = 0.5,
    ):
        self.enabled = enabled
        self.version = PAPOVersion(version)
        self.mask_ratio = mask_ratio
        self.use_noise = use_noise
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.kl_penalty = kl_penalty
        self.routing_threshold = routing_threshold
    
    def __repr__(self):
        return (
            f"PAPOConfig(enabled={self.enabled}, version={self.version.value}, "
            f"mask_ratio={self.mask_ratio}, kl_coef={self.kl_coef}, "
            f"entropy_coef={self.entropy_coef})"
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PAPOConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames})
