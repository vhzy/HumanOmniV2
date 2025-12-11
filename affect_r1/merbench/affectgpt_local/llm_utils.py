"""LLM-based helpers mirroring AffectGPT's ew_metric module."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .common import get_active_config
from .utils import iter_batches

DEFAULT_BATCH = 8


def _postprocess(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("输入", "输出", "翻译", "让我们来翻译一下：", "output", "Output", "input", "Input"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
    if cleaned.startswith(":") or cleaned.startswith("："):
        cleaned = cleaned[1:].lstrip()
    return cleaned.replace("\n", "").strip()


def _batch_generate(llm: LLM, tokenizer, sampling_params: SamplingParams, prompts: Sequence[str]) -> List[str]:
    message_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(texts, sampling_params)
    responses = []
    for output in outputs:
        response = output.outputs[0].text
        responses.append(_postprocess(response))
    return responses


def _prompts_reason_to_openset(reasons: Sequence[str]) -> List[str]:
    template = (
        "Please assume the role of an expert in the field of emotions. "
        "We provide clues that may be related to the emotions of the characters. "
        "Based on the provided clues, please identify the emotional states of the main character. "
        "The main character is the one with the most detailed clues. "
        "Please separate different emotional categories with commas and output only the clearly "
        "identifiable emotional categories in a list format. "
        "If none are identified, please output an empty list. "
        "Input: We cannot recognize his emotional state; Output: [] "
        "Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] "
        "Input: {reason}; Output: "
    )
    return [template.format(reason=reason) for reason in reasons]


def _prompts_openset_to_sentiment(opensets: Sequence[str]) -> List[str]:
    template = (
        "Please act as an expert in the field of emotions. "
        "We provide a few words to describe the emotions of a character. "
        "Please choose the most likely sentiment from the given candidates: [positive, negative, neutral] "
        "Please direct output answer without analyzing process. "
        "Constraints:"
        "1. Output ONLY the label. Do not analyze or explain."
        "2. Do not use words like 'mixed', 'ambiguous', or 'none'. You must select the best fit from the three candidates."
        "3. If the input is empty, output 'neutral'."
        "Input: [joyful]; Output: positive "
        "Input: []; Output: neutral "
        "Input: {openset}; Output: "
    )
    return [template.format(openset=openset) for openset in opensets]


def func_read_batch_calling_model(modelname: str = "Qwen25") -> Tuple[LLM, AutoTokenizer, SamplingParams]:
    cfg = get_active_config()
    model_path = cfg.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=512,
    )
    return llm, tokenizer, sampling_params


def extract_openset_batchcalling(
    *,
    name2reason: Dict[str, str],
    llm: LLM | None = None,
    tokenizer=None,
    sampling_params: SamplingParams | None = None,
    modelname: str = "Qwen25",
) -> Tuple[List[str], List[str]]:
    if llm is None or tokenizer is None or sampling_params is None:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname)
    names = list(name2reason.keys())
    responses = []
    for batch_names in iter_batches(names, DEFAULT_BATCH):
        reasons = [name2reason[name] for name in batch_names]
        prompts = _prompts_reason_to_openset(reasons)
        batch_responses = _batch_generate(llm, tokenizer, sampling_params, prompts)
        responses.extend(batch_responses)
    return names, responses


def openset_to_sentiment_batchcalling(
    *,
    name2openset: Dict[str, str],
    llm: LLM | None = None,
    tokenizer=None,
    sampling_params: SamplingParams | None = None,
    modelname: str = "Qwen25",
) -> Tuple[List[str], List[str]]:
    if llm is None or tokenizer is None or sampling_params is None:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname)
    names = list(name2openset.keys())
    responses = []
    for batch_names in iter_batches(names, DEFAULT_BATCH):
        opensets = [name2openset[name] for name in batch_names]
        prompts = _prompts_openset_to_sentiment(opensets)
        batch_responses = _batch_generate(llm, tokenizer, sampling_params, prompts)
        responses.extend(batch_responses)
    return names, responses

