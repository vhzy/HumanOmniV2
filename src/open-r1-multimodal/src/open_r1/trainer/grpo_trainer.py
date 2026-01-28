# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import datasets
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    # AriaForConditionalGeneration,
    # AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    # Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available, is_rich_available,  is_datasets_available
from transformers.trainer_utils import seed_worker
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, print_prompt_completions_sample
from trl import GRPOTrainer
from trl.import_utils import is_deepspeed_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import torch.distributed as dist


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule

# Logit-based reward (optional)
# 动态添加 affect_r1 的父目录到 sys.path（解决 torchrun 子进程不继承 PYTHONPATH 的问题）
# 注意：需要添加 affect_r1 的父目录，这样 Python 才能找到 affect_r1 包
import sys
import os as _os
_affect_r1_parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../../../.."))  # HumanOmniV2 目录
if _affect_r1_parent not in sys.path:
    sys.path.insert(0, _affect_r1_parent)

try:
    from affect_r1.logit_reward import (
        LogitRewardConfig,
        LogitRewardComputer,
        create_logit_reward_computer,
    )
    LOGIT_REWARD_AVAILABLE = True
except ImportError:
    LOGIT_REWARD_AVAILABLE = False

# PAPO (Perception-Aware Policy Optimization) support
try:
    from affect_r1.papo_utils import (
        PAPOConfig,
        PAPOVersion,
        mask_all_multimodal,
        mask_visual_inputs,
        mask_audio_inputs,
        compute_papo_loss_v0,
        compute_papo_loss_v1,
        compute_papo_loss_v2,
        compute_modality_routing_mask,
        sentence_mask_to_token_mask,
    )
    PAPO_AVAILABLE = True
except ImportError:
    PAPO_AVAILABLE = False
    PAPOConfig = None

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def find_mask_between_patterns_1d(input_tensor: torch.Tensor, 
                                  start_pattern_list: list, 
                                  end_pattern_list: list) -> torch.Tensor:
    """
    (Helper function - same as before)
    Finds the mask for a single 1D tensor.
    """
    assert input_tensor.ndim == 1, "Input tensor must be 1-dimensional"
    
    device = input_tensor.device
    dtype = input_tensor.dtype # Use input tensor's dtype

    # Ensure patterns are tensors on the correct device and dtype
    start_pattern = torch.tensor(start_pattern_list, dtype=dtype, device=device)
    end_pattern = torch.tensor(end_pattern_list, dtype=dtype, device=device)

    n = input_tensor.shape[0]
    len_start = len(start_pattern)
    len_end = len(end_pattern)

    start_idx = -1
    end_idx = -1

    # --- Find start_pattern index ---
    if n >= len_start:
        start_windows = input_tensor.unfold(0, len_start, 1)
        start_matches = (start_windows == start_pattern).all(dim=1)
        start_indices = start_matches.nonzero(as_tuple=True)[0]
        if start_indices.numel() > 0:
            start_idx = start_indices[0].item() # Assume first match
        else:
            # Indicate pattern not found for this row
            return torch.zeros_like(input_tensor, dtype=torch.long, device=device) 
            # raise ValueError("Start pattern not found in the tensor.") # Original behavior
    else:
        return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Find end_pattern index ---
    if n >= len_end:
        # Search *after* the start pattern to ensure correct order if multiple end patterns exist
        # Although problem states "only one region", this adds robustness
        search_area_end = input_tensor[start_idx + len_start:] 
        if search_area_end.numel() >= len_end:
            end_windows = search_area_end.unfold(0, len_end, 1)
            end_matches = (end_windows == end_pattern).all(dim=1)
            end_indices = end_matches.nonzero(as_tuple=True)[0]
            if end_indices.numel() > 0:
                 # Index relative to the start of search_area_end, need to add offset
                relative_end_idx = end_indices[0].item()
                end_idx = start_idx + len_start + relative_end_idx 
            else:
                 # End pattern not found *after* start pattern
                return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
        else:
            # Not enough elements after start pattern to contain end pattern
             return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
    else:
       return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Calculate mask region ---
    mask_start = start_idx + len_start
    mask_end = end_idx # end_idx is the *start* of the end pattern

    # mask_start = start_idx 
    # mask_end = end_idx + len_end 

    # --- Create and fill mask ---
    mask = torch.zeros_like(input_tensor, dtype=torch.long, device=device)

    # if mask_start < mask_end:
    #     mask[mask_start:-1] = 1
    if mask_start < mask_end:
        mask[:mask_end] = 1

    # if mask_start < mask_end:
    #     mask[mask_start:mask_end] = 1
    # else: patterns adjacent or end before start, mask remains zero, no warning needed here

    return mask



def generate_2d_mask(input_tensor_2d: torch.Tensor, 
                       start_pattern_list: list, 
                       end_pattern_list: list) -> torch.Tensor:
    """
    Generates a 2D mask by applying the 1D pattern finding logic to each row.

    Args:
        input_tensor_2d: The input 2D PyTorch Tensor (Batch x SequenceLength).
        start_pattern_list: The start pattern list.
        end_pattern_list: The end pattern list.

    Returns:
        A 2D mask tensor of the same shape as input_tensor_2d (dtype=torch.long),
        where each row's mask is generated based on the patterns found in that row.
        Rows where patterns are not found (or order is wrong) will have a mask of all zeros.
    """
    assert input_tensor_2d.ndim == 2, "Input tensor must be 2-dimensional"
    
    num_rows = input_tensor_2d.shape[0]
    if num_rows == 0:
        return torch.empty_like(input_tensor_2d, dtype=torch.long) # Handle empty input

    row_masks = []
    for i in range(num_rows):
        current_row = input_tensor_2d[i]
        # Call the 1D function for the current row
        # Modify 1D function to return zeros instead of raising error if pattern not found
        mask_1d = find_mask_between_patterns_1d(current_row, start_pattern_list, end_pattern_list)
        row_masks.append(mask_1d)

    # Stack the generated 1D masks along the batch dimension (dim=0)
    mask_2d = torch.stack(row_masks, dim=0)
    
    return mask_2d

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        # model = model.thinker # for qwen-omni

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled() or is_deepspeed_available():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)


        total_trainable_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                # print(f'train param: {name}')
                total_trainable_params += p.numel()


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None  # TODO
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.markov_reward = args.markov_reward


     
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
            renormalize_logits=True,  # 防止 softmax 溢出导致 inf/nan
        )
        if hasattr(self.vlm_module, "get_eos_token_id"):  # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))

        # 禁止在 completion 中生成多模态占位符，避免二次 forward 时触发 Qwen Omni 的多模态解析错误
        try:
            tokenizer = getattr(processing_class, "tokenizer", processing_class)
            bad_tokens = []

            # 来自 tokenizer 的多模态特殊 token（如 <image> / <video> / <audio>）
            for token_attr in ["image_token", "video_token", "audio_token"]:
                token = getattr(tokenizer, token_attr, None)
                if token is not None:
                    tid = tokenizer.convert_tokens_to_ids(token)
                    if tid is not None and tid != tokenizer.pad_token_id:
                        bad_tokens.append([int(tid)])

            # 来自模型 config 的多模态相关 token id / index
            config = model.config
            for cfg_attr in [
                "vision_start_token_id",
                "vision_end_token_id",
                "vision_token_id",
                "audio_start_token_id",
                "audio_end_token_id",
                "audio_token_id",
                "image_token_id",
                "audio_token_index",
                "image_token_index",
                "video_token_index",
            ]:
                if hasattr(config, cfg_attr):
                    tid = getattr(config, cfg_attr)
                    if tid is not None:
                        bad_tokens.append([int(tid)])

            if len(bad_tokens) > 0:
                if self.generation_config.bad_words_ids is None:
                    self.generation_config.bad_words_ids = bad_tokens
                else:
                    self.generation_config.bad_words_ids = (
                        self.generation_config.bad_words_ids + bad_tokens
                    )
        except Exception:
            # 如果构造 bad_words_ids 失败，不影响训练主体逻辑
            pass

        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon


        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # Initialize logit-based reward computer (optional)
        # Now configured via reward_funcs: logit_reward.coherence, logit_reward.perception
        self.logit_reward_computer = None
        self.logit_reward_weights = {}  # {'coherence': weight, 'perception': weight}
        self.logit_reward_scale_method = "tanh"  # 'tanh' or 'clip'
        self.logit_reward_use_neg_contrast = True  # 默认值
        
        # Get logit_reward_config from args (set by grpo_qwenomni.py)
        logit_reward_config = getattr(args, 'logit_reward_config', {})
        self.use_logit_reward = len(logit_reward_config) > 0
        
        print(f"[LogitReward DEBUG] logit_reward_config: {logit_reward_config}")
        print(f"[LogitReward DEBUG] use_logit_reward: {self.use_logit_reward}")
        print(f"[LogitReward DEBUG] LOGIT_REWARD_AVAILABLE: {LOGIT_REWARD_AVAILABLE}")
        
        if self.use_logit_reward:
            if not LOGIT_REWARD_AVAILABLE:
                warnings.warn(
                    "logit_reward.* specified in reward_funcs but logit_reward module is not available. "
                    "Please ensure affect_r1.logit_reward is in PYTHONPATH."
                )
                self.use_logit_reward = False
            else:
                # Parse logit_reward_config
                # coherence weight -> use_coherence_reward and its weight
                # perception weight -> use_perception_reward and its weight
                self.logit_reward_weights = {k: v for k, v in logit_reward_config.items() if k not in ['scale_method']}
                self.logit_reward_scale_method = logit_reward_config.get('scale_method', 'tanh')
                
                tokenizer = getattr(processing_class, 'tokenizer', processing_class)
                # 解析 use_neg_contrast 参数
                self.logit_reward_use_neg_contrast = str(getattr(args, 'logit_reward_use_neg_contrast', 'true')).lower() in ('true', '1', 'yes')
                logit_config = LogitRewardConfig(
                    use_coherence_reward='coherence' in logit_reward_config,
                    use_perception_reward='perception' in logit_reward_config,
                    alpha=1.0,  # 内部不再加权，权重在 grpo_trainer 中通过 reward_weights 统一应用
                    beta=1.0,   # 内部不再加权，权重在 grpo_trainer 中通过 reward_weights 统一应用
                    normalize_rewards=getattr(args, 'logit_reward_normalize', False),
                    use_neg_contrast=self.logit_reward_use_neg_contrast,
                )
                self.logit_reward_computer = create_logit_reward_computer(
                    tokenizer=tokenizer,
                    emotion_wheel_root=getattr(args, 'emotion_wheel_root', None),
                    config=logit_config,
                )
                print(f"[LogitReward] Initialized with unified config:")
                print(f"  - coherence: {'enabled' if 'coherence' in logit_reward_config else 'disabled'}, weight={logit_reward_config.get('coherence', 0.0)}")
                print(f"  - perception: {'enabled' if 'perception' in logit_reward_config else 'disabled'}, weight={logit_reward_config.get('perception', 0.0)}")
                print(f"  - scale_method: {self.logit_reward_scale_method}")
                print(f"  - use_neg_contrast: {logit_config.use_neg_contrast} (raw arg value: {getattr(args, 'logit_reward_use_neg_contrast', 'NOT_FOUND')})")
                if not logit_config.use_neg_contrast:
                    print(f"  - reward_scaling: raw_reward / 5.0 → {self.logit_reward_scale_method}()")

        # Initialize PAPO (Perception-Aware Policy Optimization)
        # Get PAPO config from args (set by grpo_qwenomni.py or config file)
        papo_config = getattr(args, 'papo_config', {})
        self.use_papo = papo_config.get('enabled', False) if papo_config else False
        
        print(f"[PAPO DEBUG] papo_config: {papo_config}")
        print(f"[PAPO DEBUG] use_papo: {self.use_papo}")
        print(f"[PAPO DEBUG] PAPO_AVAILABLE: {PAPO_AVAILABLE}")
        
        if self.use_papo:
            if not PAPO_AVAILABLE:
                warnings.warn(
                    "PAPO is enabled but papo_utils module is not available. "
                    "Please ensure affect_r1.papo_utils is in PYTHONPATH."
                )
                self.use_papo = False
                self.papo_config = None
            else:
                self.papo_config = PAPOConfig.from_dict(papo_config)
                print(f"[PAPO] Initialized: {self.papo_config}")
        else:
            self.papo_config = None

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                # print(inputs)
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

        # # Simple pass-through, just like original
        # return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # print(inputs)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)[0]
        use_audio_in_video = inputs[0].get("use_audio_in_video", False)
        # print(prompts_text)
        images, videos, audios = [], [], []


        for each in inputs:
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None
        # 在调用 Processor 之前


        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=False,
        )
        # Processor 未显式返回 audio_feature_lengths 时，仅在存在音频特征的前提下，
        # 借助 feature_attention_mask 推导补全，避免无音频样本产生伪造的长度信息
        if (
            "input_features" in prompt_inputs
            and prompt_inputs["input_features"] is not None
            and "audio_feature_lengths" not in prompt_inputs
            and "feature_attention_mask" in prompt_inputs
        ):
            fam = prompt_inputs["feature_attention_mask"]
            input_lengths = (fam.sum(dim=1) - 1) // 2 + 1
            prompt_inputs["audio_feature_lengths"] = (input_lengths - 2) // 2 + 1
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs["use_audio_in_video"] = use_audio_in_video

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        try:
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                    generation_config=self.generation_config
                )
                prompt_length = prompt_ids.size(1)
                if not self.vlm_module.is_embeds_input():
                    prompt_completion_ids = generate_returned_result
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                else:
                    # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                    # So the returned result of the `generate` method only contains the completion ids
                    completion_ids = generate_returned_result
                    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        except Exception as e:
            # Print problematic sample info for debugging
            print(f"\n[Error] Exception during generate(): {e}")
            print(f"[Error] Failed inputs info (batch_size={len(inputs)}, use_audio_in_video={use_audio_in_video}):")
            for i, example in enumerate(inputs):
                print(f"--- Sample {i} ---")
                if isinstance(example, dict):
                    try:
                        print(f"Keys: {sorted(example.keys())}")
                    except Exception:
                        pass
                    if "path" in example:
                        print(f"Path: {example.get('path')}")
                    if "audio_path" in example:
                        print(f"Audio Path: {example.get('audio_path')}")
                    if "prompt" in example:
                        # Print first 400 chars of prompt or conversation
                        prompt_preview = str(example.get("prompt"))[200:]
                        print(f"Prompt preview: {prompt_preview}...")
                else:
                    print(f"Example type: {type(example).__name__}")
                    print(f"Example preview: {str(example)[200:]}...")
                # prepared prompt often easiest to search in logs
                if isinstance(prompts_text, (list, tuple)) and i < len(prompts_text):
                    prepared_preview = str(prompts_text[i])[200:]
                    print(f"Prepared prompt preview: {prepared_preview}...")
            raise


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        # multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        excluded_keys = set()  # 保留全部多模态特征，避免音视频特征被丢弃
        multimodal_inputs = {k: prompt_inputs[k] for k in multimodal_keywords if k in prompt_inputs and k not in excluded_keys}
        try:
            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                    old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
                else:
                    old_per_token_logps = None

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                        )
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        except Exception as e:
            # Print problematic sample info for debugging
            print(f"\n[Error] Exception during logit computation: {e}")
            print(f"[Error] Failed inputs info (batch_size={len(inputs)}, use_audio_in_video={use_audio_in_video}):")
            for i, example in enumerate(inputs):
                print(f"--- Sample {i} ---")
                if isinstance(example, dict):
                    try:
                        print(f"Keys: {sorted(example.keys())}")
                    except Exception:
                        pass
                    if "path" in example:
                        print(f"Path: {example.get('path')}")
                    if "audio_path" in example:
                        print(f"Audio Path: {example.get('audio_path')}")
                    if "prompt" in example:
                        # Print first 400 chars of prompt or conversation
                        prompt_preview = str(example.get("prompt"))[200:]
                        print(f"Prompt preview: {prompt_preview}...")
                else:
                    print(f"Example type: {type(example).__name__}")
                    print(f"Example preview: {str(example)[200:]}...")
                if isinstance(prompts_text, (list, tuple)) and i < len(prompts_text):
                    prepared_preview = str(prompts_text[i])[200:]
                    print(f"Prepared prompt preview: {prepared_preview}...")
            raise


        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
        # else branch intentionally omitted in official logic

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
        # For PAPO V2: collect similarity matrices from reward functions
        papo_v2_sim_results = None
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                
                # Check if reward function returned similarity matrices (for PAPO V2)
                # rubric_perc_reward_with_matrices returns (rewards, sim_results)
                if isinstance(output_reward_func, tuple) and len(output_reward_func) == 2:
                    output_reward_func, sim_results = output_reward_func
                    # Store sim_results for PAPO V2 (only from the perception reward)
                    if self.use_papo and self.papo_config and self.papo_config.version == PAPOVersion.V2:
                        func_name = reward_func.__name__
                        if 'perc' in func_name.lower() or 'perception' in func_name.lower():
                            papo_v2_sim_results = sim_results
                            print(f"[PAPO V2] Captured similarity matrices from {func_name}, count: {len(sim_results)}")
                
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # 可选：格式守门，format 错则该样本奖励全归零
        if getattr(self.args, "format_gate_all", False):
            format_cols = [idx for idx, f in enumerate(self.reward_funcs) if getattr(f, "__name__", "") == "format_reward"]
            if len(format_cols) > 0:
                format_ok = torch.ones(len(prompts), device=device, dtype=torch.bool)
                for col in format_cols:
                    format_ok = format_ok & (rewards_per_func[:, col] > 0.5)
                rewards_per_func = torch.where(
                    format_ok.unsqueeze(1),
                    rewards_per_func,
                    torch.zeros_like(rewards_per_func),
                )

        # 可选：关闭 stage2 奖励（format / perception / coherence），但仍保留 PAPO V2 的相似度路由
        if getattr(self.args, "disable_stage2_rewards", False):
            stage2_names = {"rubric_perc_reward_with_matrices", "rubric_perc_reward", "rubric_coh_reward"}
            for idx, reward_func in enumerate(self.reward_funcs):
                if getattr(reward_func, "__name__", "") in stage2_names:
                    rewards_per_func[:, idx] = 0

        # markov
        if rewards_per_func.size(1) ==4 and self.markov_reward: # format, acc, reason, evi
            print("using markov")
            not_valid_evidence_index = rewards_per_func[:, -1]<=0.4
            not_valid_reason_index = rewards_per_func[:, -2]<=0.4
            rewards_per_func[not_valid_evidence_index, 1] = 0
            rewards_per_func[not_valid_evidence_index, 2] = 0
            rewards_per_func[not_valid_reason_index, 1] = 0

            # not_valid_format_index = rewards_per_func[:, 1]<=0.2
            # rewards_per_func[not_valid_format_index, 0] = 0

        # if rewards_per_func.size(1) ==2: # format, acc, reason, evi
        #     print("using markov")
        # not_valid_format_index = rewards_per_func[:, 1]<=0.2
        
        # rewards_per_func[not_valid_format_index, :] = 0

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)

        def compute_advantage(rewards):
            # Compute grouped-wise rewards
            # Each group consists of num_generations completions for the same prompt
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            
            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            return advantages, std_grouped_rewards

        full_rewards = []
        patial_rewards = []
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.__name__
            if "patial" not in func_name:
                full_rewards.append(rewards_per_func[:, i] * self.reward_weights[i].to(device))
            else:
                patial_rewards.append((func_name, rewards_per_func[:, i] * self.reward_weights[i].to(device)))
        
        
        
        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

         # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        # rewards_per_func = rewards_per_func[process_slice, :]

        rewards = torch.stack(full_rewards, dim=-1).nansum(dim=1)
        
        # Compute logit-based rewards (optional) - 在归一化之前添加到 rewards
        logit_rewards_dict = None
        if self.use_logit_reward and self.logit_reward_computer is not None:
            print(f"[LogitReward DEBUG] Computing logit rewards...")
            try:
                # 准备 GT 类别列表（从 inputs 中获取 openset）
                gt_categories_list = []
                for example in inputs:
                    openset = example.get('openset', [])
                    if isinstance(openset, str):
                        openset = [s.strip() for s in openset.split(',')]
                    elif openset is None:
                        openset = []
                    gt_categories_list.append(openset if openset else [])
                
                # 计算 logit reward
                logit_rewards_dict = self.logit_reward_computer.compute_rewards(
                    model=self.model,
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    multimodal_inputs=multimodal_inputs,
                    gt_categories_list=gt_categories_list,
                    completion_ids=completion_ids,
                )
                
                # 使用统一的权重配置，手动加权各个组件
                # logit_reward_weights: {'coherence': weight, 'perception': weight}
                # 重要：logit_rewards_dict 是用本地数据计算的，维度是 (local_batch_size,)
                # 需要用本地维度初始化 logit_total_reward，然后 gather 后再加到 rewards
                local_batch_size = logit_rewards_dict["coherence"].size(0) if "coherence" in logit_rewards_dict else logit_rewards_dict["perception"].size(0)
                logit_total_reward = torch.zeros(local_batch_size, device=device)
                
                # 根据 scale_method 选择缩放方式
                # tanh: 平滑映射到 [-1, 1]
                # clip: 硬截断到 [-2, 2]
                # 特别处理：当 use_neg_contrast=False 时，先除以 5 再应用 tanh
                def scale_reward(raw_reward, method, use_neg_contrast):
                    # 如果不使用 neg contrast，先缩小 5 倍
                    # if not use_neg_contrast:
                    #     if raw_reward < 0:
                    #         raw_reward = raw_reward / 2.0
                    
                    if method == 'clip':
                        return torch.clamp(raw_reward, -5.0, 5.0)
                    else:  # 默认 tanh
                        return torch.tanh(raw_reward)
                
                scale_method = self.logit_reward_scale_method
                use_neg_contrast = getattr(self, 'logit_reward_use_neg_contrast', True)
                
                if 'coherence' in self.logit_reward_weights:
                    coherence_weight = self.logit_reward_weights['coherence']
                    coherence_raw = logit_rewards_dict["coherence"]
                    coherence_scaled = scale_reward(coherence_raw, scale_method, use_neg_contrast)
                    logit_total_reward = logit_total_reward + coherence_weight * coherence_scaled
                    scaling_info = f"/5 then {scale_method}" if not use_neg_contrast else scale_method
                    print(f"[LogitReward DEBUG] coherence: weight={coherence_weight}, raw_mean={coherence_raw.mean().item():.4f}, {scaling_info}_mean={coherence_scaled.mean().item():.4f}")
                
                if 'perception' in self.logit_reward_weights:
                    perception_weight = self.logit_reward_weights['perception']
                    perception_raw = logit_rewards_dict["perception"]
                    perception_scaled = scale_reward(perception_raw, scale_method, use_neg_contrast)
                    logit_total_reward = logit_total_reward + perception_weight * perception_scaled
                    scaling_info = f"/5 then {scale_method}" if not use_neg_contrast else scale_method
                    print(f"[LogitReward DEBUG] perception: weight={perception_weight}, raw_mean={perception_raw.mean().item():.4f}, {scaling_info}_mean={perception_scaled.mean().item():.4f}")
                
                # 将加权后的 logit reward 添加到 rewards 中
                # 重要：先 gather logit_total_reward 以匹配 rewards 的维度
                # rewards 已经是 gathered 后的全局数据，维度是 (N * num_gpus,)
                # logit_total_reward 是本地数据，维度是 (local_batch_size,)
                logit_total_reward = self.accelerator.gather(logit_total_reward)
                rewards = rewards + logit_total_reward
                print(f"[LogitReward DEBUG] Added logit reward to rewards (after gather), local_batch={local_batch_size}, gathered_size={logit_total_reward.size(0)}, total_mean: {logit_total_reward.mean().item():.4f}")
                
            except Exception as e:
                import traceback
                print(f"[LogitReward ERROR] Failed to compute logit rewards: {e}")
                traceback.print_exc()
                logit_rewards_dict = None
        
        # 现在一起归一化（包含 logit reward）
        advantages, std_grouped_rewards = compute_advantage(rewards)
        advantages = advantages[process_slice]

        patial_advantages = []
        
        if len(patial_rewards) > 0:
            for func_name, partial_reward in patial_rewards:
                partial_reward = compute_advantage(rewards)[0][process_slice]
                # evidence  : 27, 68, 27480 ... 522, 68 27480
                # think  : 13708, 766,  ... 522, 26865
                # if "evidence" in func_name:
                #     mask = generate_2d_mask(completion_ids, [27, 68, 27480], [522, 68, 27480])
                if "context" in func_name: # context
                    mask = generate_2d_mask(completion_ids, [34528], [522, 2147])
                else: # logical
                    mask = generate_2d_mask(completion_ids, [13708, 766], [522, 26865])
                patial_advantages.append({"name": func_name, "reward":partial_reward, "mask": mask})

        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        # Log logit-based rewards if computed
        if logit_rewards_dict is not None:
            if "coherence" in logit_rewards_dict:
                self._metrics[mode]["rewards/logit_coherence"].append(
                    logit_rewards_dict["coherence"].mean().item()
                )
            if "perception" in logit_rewards_dict:
                self._metrics[mode]["rewards/logit_perception"].append(
                    logit_rewards_dict["perception"].mean().item()
                )
            if "total" in logit_rewards_dict:
                self._metrics[mode]["rewards/logit_total"].append(
                    logit_rewards_dict["total"].mean().item()
                )
            # 同时记录两种 use_neg_contrast 模式的 coherence reward 曲线（用于对比）
            if "coherence_contrast" in logit_rewards_dict:
                self._metrics[mode]["rewards/logit_coherence_contrast"].append(
                    logit_rewards_dict["coherence_contrast"].mean().item()
                )
            if "coherence_no_contrast" in logit_rewards_dict:
                self._metrics[mode]["rewards/logit_coherence_no_contrast"].append(
                    logit_rewards_dict["coherence_no_contrast"].mean().item()
                )

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # Gather across processes so lengths对齐
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = {"sum": rewards.tolist()}
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, nn.Module):
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                rewards_to_log[reward_func_name] = rewards_per_func[:, i].tolist()
            advantages_to_log = gather_object(advantages.tolist())
            
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        advantages=advantages_to_log,
                        step=self.state.global_step,
                    )
                # if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                #     import pandas as pd

                #     # For logging
                #     table = {
                #         "step": [str(self.state.global_step)] * len(rewards),
                #         "prompt": prompts_to_log,
                #         "completion": completions_to_log,
                #         "reward": rewards.tolist(),
                #     }
                #     df = pd.DataFrame(table)
                #     wandb.log({"completions": wandb.Table(dataframe=df)})

        # ===================== PAPO: Compute masked forward passes =====================
        papo_data = {}
        if self.use_papo and self.papo_config is not None:
            try:
                with torch.no_grad():
                    prompt_length = prompt_ids.size(1)
                    
                    if self.papo_config.version == PAPOVersion.V0:
                        # V0: Mask both audio and video simultaneously
                        masked_inputs = mask_all_multimodal(
                            multimodal_inputs, 
                            mask_ratio=self.papo_config.mask_ratio,
                            noise=self.papo_config.use_noise
                        )
                        masked_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **masked_inputs
                        )
                        masked_per_token_logps = masked_per_token_logps[:, prompt_length - 1:]
                        papo_data["masked_per_token_logps"] = masked_per_token_logps
                        print(f"[PAPO V0] Computed masked logps, shape: {masked_per_token_logps.shape}")
                        
                    elif self.papo_config.version == PAPOVersion.V1:
                        # V1: Mask visual and audio separately
                        # No visual input
                        no_v_inputs = mask_visual_inputs(
                            multimodal_inputs,
                            mask_ratio=self.papo_config.mask_ratio,
                            noise=self.papo_config.use_noise
                        )
                        no_v_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **no_v_inputs
                        )
                        no_v_per_token_logps = no_v_per_token_logps[:, prompt_length - 1:]
                        papo_data["no_v_per_token_logps"] = no_v_per_token_logps
                        
                        # No audio input
                        no_a_inputs = mask_audio_inputs(
                            multimodal_inputs,
                            mask_ratio=self.papo_config.mask_ratio,
                            noise=self.papo_config.use_noise
                        )
                        no_a_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **no_a_inputs
                        )
                        no_a_per_token_logps = no_a_per_token_logps[:, prompt_length - 1:]
                        papo_data["no_a_per_token_logps"] = no_a_per_token_logps
                        print(f"[PAPO V1] Computed no_v and no_a logps")
                        
                    elif self.papo_config.version == PAPOVersion.V2:
                        # V2: Matrix-guided fine-grained PAPO
                        # First compute the masked logps (same as V1)
                        no_v_inputs = mask_visual_inputs(
                            multimodal_inputs,
                            mask_ratio=self.papo_config.mask_ratio,
                            noise=self.papo_config.use_noise
                        )
                        no_v_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **no_v_inputs
                        )
                        no_v_per_token_logps = no_v_per_token_logps[:, prompt_length - 1:]
                        papo_data["no_v_per_token_logps"] = no_v_per_token_logps
                        
                        no_a_inputs = mask_audio_inputs(
                            multimodal_inputs,
                            mask_ratio=self.papo_config.mask_ratio,
                            noise=self.papo_config.use_noise
                        )
                        no_a_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, **no_a_inputs
                        )
                        no_a_per_token_logps = no_a_per_token_logps[:, prompt_length - 1:]
                        papo_data["no_a_per_token_logps"] = no_a_per_token_logps
                        
                        # V2 routing masks will be computed in compute_loss using sim_matrices from reward
                        # For now, store completions for sentence extraction
                        papo_data["completions_text"] = completions_text
                        print(f"[PAPO V2] Computed no_v and no_a logps (routing masks will be in compute_loss)")
                        
            except Exception as e:
                import traceback
                print(f"[PAPO ERROR] Failed to compute masked logps: {e}")
                traceback.print_exc()
                papo_data = {}
        # ===================== END PAPO =====================
        
        # ===================== PAPO V2: Convert sim_results to token masks =====================
        sim_matrices = {}
        if self.use_papo and self.papo_config and self.papo_config.version == PAPOVersion.V2:
            if papo_v2_sim_results is not None and len(papo_v2_sim_results) > 0:
                try:
                    # Import the conversion function from affect_reward
                    from affect_reward import compute_modality_token_masks
                    
                    token_masks_dict = compute_modality_token_masks(
                        sim_results=papo_v2_sim_results,
                        completion_ids=completion_ids,
                        tokenizer=self.processing_class.tokenizer,
                        threshold=self.papo_config.routing_threshold,
                    )
                    
                    sim_matrices = {
                        'visual_token_mask': token_masks_dict['visual_token_mask'],
                        'audio_token_mask': token_masks_dict['audio_token_mask'],
                    }
                    
                    print(f"[PAPO V2] Converted sim_results to token masks, shapes: "
                          f"visual={sim_matrices['visual_token_mask'].shape}, "
                          f"audio={sim_matrices['audio_token_mask'].shape}")
                    
                    # Log routing statistics
                    visual_ratio = sim_matrices['visual_token_mask'].sum() / sim_matrices['visual_token_mask'].numel()
                    audio_ratio = sim_matrices['audio_token_mask'].sum() / sim_matrices['audio_token_mask'].numel()
                    print(f"[PAPO V2] Token routing: visual={visual_ratio:.2%}, audio={audio_ratio:.2%}")
                    
                except Exception as e:
                    import traceback
                    print(f"[PAPO V2 ERROR] Failed to convert sim_results to token masks: {e}")
                    traceback.print_exc()
                    sim_matrices = {}
            else:
                print(f"[PAPO V2 WARNING] No sim_results available, will fallback to V1 behavior")
        # ===================== END PAPO V2 =====================

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "patial_advantages": patial_advantages,
            "papo_data": papo_data,
            "sim_matrices": sim_matrices,  # Add sim_matrices for PAPO V2
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")


    
        # Check if we need to generate new completions or use buffered ones
        # if self.state.global_step % self.num_iterations == 0:
        #     inputs = self._generate_and_score_completions(inputs)
        #     self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        # else:
        #     inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        # self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]
        patial_advantages = inputs["patial_advantages"]

        for patial_advantage in patial_advantages:
            advantages = advantages + patial_advantage["reward"]*patial_advantage["mask"] 


        mode = "eval" if self.control.should_evaluate else "train"
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            mean_kl_for_threshold = self.accelerator.gather_for_metrics(mean_kl.detach()).mean().item()
            self._metrics[mode]["kl"].append(mean_kl_for_threshold)
            
            if self.state.global_step > (self.state.max_steps / 2):
                beta = self.beta * 0.25
            else:
                beta = self.beta * (1 - 0.75 * self.state.global_step / (self.state.max_steps / 2))

            # If KL is too high, restore beta to the initial value (self.beta)
            # if mean_kl_for_threshold > 0.08:
            #     beta = self.beta

            per_token_loss = per_token_loss + beta * per_token_kl
            self._metrics[mode]["beta"].append(self.accelerator.gather_for_metrics(torch.tensor(beta, device=per_token_logps.device)).mean().item())

        # Compute final loss
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log clip ratio
        is_clipped = (per_token_loss2 < per_token_loss1).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        # ===================== PAPO Loss =====================
        if self.use_papo and self.papo_config is not None:
            papo_data = inputs.get("papo_data", {})
            
            if papo_data:
                try:
                    if self.papo_config.version == PAPOVersion.V0:
                        # V0: Both modalities masked together
                        masked_logps = papo_data.get("masked_per_token_logps")
                        if masked_logps is not None:
                            papo_loss, papo_metrics = compute_papo_loss_v0(
                                log_probs_full=per_token_logps,
                                log_probs_masked=masked_logps,
                                completion_mask=completion_mask,
                                kl_coef=self.papo_config.kl_coef,
                                entropy_coef=self.papo_config.entropy_coef,
                                kl_penalty=self.papo_config.kl_penalty,
                            )
                            loss = loss + papo_loss
                            for k, v in papo_metrics.items():
                                self._metrics[mode][k].append(v)
                                
                    elif self.papo_config.version == PAPOVersion.V1:
                        # V1: Separate visual and audio masking
                        no_v_logps = papo_data.get("no_v_per_token_logps")
                        no_a_logps = papo_data.get("no_a_per_token_logps")
                        if no_v_logps is not None and no_a_logps is not None:
                            papo_loss, papo_metrics = compute_papo_loss_v1(
                                log_probs_full=per_token_logps,
                                log_probs_no_v=no_v_logps,
                                log_probs_no_a=no_a_logps,
                                completion_mask=completion_mask,
                                kl_coef=self.papo_config.kl_coef,
                                entropy_coef=self.papo_config.entropy_coef,
                                kl_penalty=self.papo_config.kl_penalty,
                            )
                            loss = loss + papo_loss
                            for k, v in papo_metrics.items():
                                self._metrics[mode][k].append(v)
                                
                    elif self.papo_config.version == PAPOVersion.V2:
                        # V2: Matrix-guided fine-grained PAPO
                        no_v_logps = papo_data.get("no_v_per_token_logps")
                        no_a_logps = papo_data.get("no_a_per_token_logps")
                        
                        # Get similarity matrices from inputs (computed by reward functions)
                        sim_matrices = inputs.get("sim_matrices", {})
                        
                        if no_v_logps is not None and no_a_logps is not None:
                            # If we have similarity matrices, use them for routing
                            if sim_matrices:
                                visual_token_mask = sim_matrices.get("visual_token_mask", 
                                    torch.ones_like(completion_mask))
                                audio_token_mask = sim_matrices.get("audio_token_mask",
                                    torch.ones_like(completion_mask))
                            else:
                                # Fallback: use all tokens (same as V1)
                                visual_token_mask = torch.ones_like(completion_mask)
                                audio_token_mask = torch.ones_like(completion_mask)
                            
                            papo_loss, papo_metrics = compute_papo_loss_v2(
                                log_probs_full=per_token_logps,
                                log_probs_no_v=no_v_logps,
                                log_probs_no_a=no_a_logps,
                                completion_mask=completion_mask,
                                visual_token_mask=visual_token_mask,
                                audio_token_mask=audio_token_mask,
                                kl_coef=self.papo_config.kl_coef,
                                entropy_coef=self.papo_config.entropy_coef,
                                kl_penalty=self.papo_config.kl_penalty,
                            )
                            loss = loss + papo_loss
                            for k, v in papo_metrics.items():
                                self._metrics[mode][k].append(v)
                                
                except Exception as e:
                    import traceback
                    print(f"[PAPO ERROR] Failed to compute PAPO loss: {e}")
                    traceback.print_exc()
        # ===================== END PAPO Loss =====================

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=train_dataset if train_dataset is not None else self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
