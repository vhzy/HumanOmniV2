"""
Counterfactual Hallucination Detection (CHAIR-M for Multimodal Emotion Recognition)

核心思想：
1. 读取 counterfactual 推理结果（如 mask_audio 的 checkpoint-3000.jsonl）
2. 用 LLM（GPT 或 Qwen）从 <think> 内容中提取多模态线索：
   - visual_clues: 视觉线索
   - audio_clues: 音频线索
   - text_clues: 文本线索（从 subtitle 统计）
3. 计算幻觉率（仅 mask 模式）：
   - 分母：N_V + N_A + N_T（总线索数）
   - 分子（mask 哪个就看哪个）：N_masked

支持三种模式：
1. mask_audio: mask 音频后检测音频幻觉
2. mask_visual: mask 视频后检测视觉幻觉
3. baseline (mask_modality=none): 不 mask，只统计线索数量作为对照组

例如 mask_audio 场景：
- 如果模型在没有音频的情况下仍然生成了音频线索，这就是幻觉
- 幻觉率 = N_A / (N_V + N_A + N_T)

Baseline 场景：
- 不计算幻觉率，只统计各模态线索数量
- 用于与 mask 实验对比，作为参考基准

支持 GPT 和 Qwen 两种推理引擎。

使用示例 - Mask 模式（计算幻觉率）：
CUDA_VISIBLE_DEVICES=0 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/check_hallucination.py \
  --input /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_baseline5/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/checkpoint-3262.jsonl\
  --output /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_baseline5/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/hallucination_results_Qwen.jsonl \
  --engine qwen \
  --llm-name Qwen25 \
  --dataset-root /mnt/afs/hanzhiyuan/MER-UniBench/data

使用示例 - Baseline 模式（只统计线索）：
CUDA_VISIBLE_DEVICES=0 python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/check_hallucination.py \
  --input /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo44_v0/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/checkpoint-3262.jsonl \
  --output /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output2/affect_r1_grpo2_papo44_v0/inference-cf/results-ovmerdplus/merbench_cf_new_10_baseline/clue_extraction_Qwen.jsonl \
  --engine qwen \
  --llm-name Qwen25 \
  --dataset-root /mnt/afs/hanzhiyuan/MER-UniBench/data

使用 GPT（并发处理，更快）：
python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/check_hallucination.py \
  --input /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_stage2_13/inference-cf/results-ovmerdplus/merbench_cf10_mask_visual/checkpoint-3000.jsonl \
  --output /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/output/affect_r1_grpo_stage2_13/inference-cf/results-ovmerdplus/merbench_cf10_mask_visual/hallucination_results_Gpt.jsonl \
  --engine gpt \
  --max-workers 8

输出文件：
1. hallucination_results_{Qwen|Gpt}.jsonl / clue_extraction_{Qwen|Gpt}.jsonl - 每个样本的详细线索和指标
2. perception_hallucination_{Qwen|Gpt}.txt - 统计报告（自动保存在输入文件同目录）
"""

import json
import re
import os
import sys
import argparse
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from openai import AzureOpenAI

# ==================== 配置 ====================
# Azure OpenAI 配置
import os
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_API_KEY_HERE")
ENDPOINT = "https://duomotai.openai.azure.com/"
API_VERSION = "2025-04-01-preview"
MODEL_NAME = "gpt-5"

# 并发配置
MAX_WORKERS = 32

# 初始化 Azure OpenAI 客户端
gpt_client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

# Qwen 模型配置（延迟加载）
qwen_llm = None
qwen_tokenizer = None
qwen_sampling_params = None

# ==================== Prompt 模板 ====================
SYSTEM_PROMPT_EXTRACT = """You are an expert linguistic annotator for a Multimodal Emotion Recognition dataset.

**Task:** Analyze the provided 'Reasoning Text' (the content inside <think> tags). Your goal is to extract the **multimodal evidence** the model used to reach the conclusion. 

**Extraction Rules:**
1. **Visual Cues:** Extract specific phrases describing facial expressions, body language, or scene details mentioned in the text.
   - Examples: "eyes closed", "serious expression", "looking down", "furrowed brows", "clenched fists"
   - Keywords to look for: "expression", "face", "body", "posture", "gesture", "looking", "eyes", "video clues"

2. **Audio Cues:** Extract specific phrases describing voice quality, tone, speed, or volume.
   - Examples: "shaky voice", "aggressive tone", "fast speaking speed", "trembling voice", "loud volume"
   - Keywords to look for: "voice", "tone", "audio", "sound", "speaking", "speech"

3. **Text Cues:** Extract phrases where the model explicitly references or quotes subtitle/caption content.
   - Examples: "subtitle says", "caption content", "the text mentions", "according to the subtitle"
   - Look for direct quotes from subtitles or references to textual content
   - If the model says "the subtitle content '...' " or "caption reads '...'", extract the quoted content

**Output Format (JSON):**
{
  "visual_cues": ["phrase 1", "phrase 2"],
  "audio_cues": ["phrase 1", "phrase 2"],
  "text_cues": ["phrase 1", "phrase 2"]
}

**Important:**
- Extract **short, semantically complete phrases** (2-6 words), not full sentences.
- Do not hallucinate. Only extract what is explicitly written in the input text.
- If a category is missing in the text, leave the list empty.
- For text_cues: only extract if the model explicitly references or quotes subtitle/caption content.
"""


# ==================== 辅助函数 ====================
def call_gpt_extract_clues(think_content: str) -> Dict[str, List[str]]:
    """调用 GPT 从 think 内容中提取多模态线索"""
    if not think_content or not think_content.strip():
        return {"visual_cues": [], "audio_cues": [], "text_cues": []}
    
    user_content = f"""**Input Text:**
{think_content}

**Instruction:**
Extract the visual, audio, and text cues from the above reasoning text. Return ONLY the JSON object.
"""
    
    try:
        response = gpt_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
                {"role": "user", "content": user_content},
            ],
            temperature=1.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return {
            "visual_cues": result.get("visual_cues", []),
            "audio_cues": result.get("audio_cues", []),
            "text_cues": result.get("text_cues", [])
        }
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        return {"visual_cues": [], "audio_cues": [], "text_cues": []}


def call_qwen_extract_clues(think_content: str) -> Dict[str, List[str]]:
    """调用 Qwen 从 think 内容中提取多模态线索"""
    global qwen_llm, qwen_tokenizer, qwen_sampling_params
    
    if not think_content or not think_content.strip():
        return {"visual_cues": [], "audio_cues": [], "text_cues": []}
    
    if qwen_llm is None:
        raise RuntimeError("Qwen model not initialized. Call initialize_qwen() first.")
    
    user_content = f"""**Input Text:**
{think_content}

**Instruction:**
Extract the visual, audio, and text cues from the above reasoning text. Return ONLY the JSON object.
"""
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
            {"role": "user", "content": user_content},
        ]
        
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = qwen_llm.generate([text], qwen_sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        # 尝试从输出中提取 JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "visual_cues": result.get("visual_cues", []),
                "audio_cues": result.get("audio_cues", []),
                "text_cues": result.get("text_cues", [])
            }
        else:
            print(f"[WARNING] Failed to parse Qwen output as JSON: {response_text[:200]}")
            return {"visual_cues": [], "audio_cues": [], "text_cues": []}
    except Exception as e:
        print(f"[ERROR] Qwen call failed: {e}")
        return {"visual_cues": [], "audio_cues": [], "text_cues": []}


def compute_hallucination_metrics(
    visual_cues: List[str],
    audio_cues: List[str],
    text_cues: List[str],
    masked_modality: str,
) -> Dict:
    """
    计算 CHAIR-M 风格的幻觉指标
    
    Args:
        visual_cues: 视觉线索列表
        audio_cues: 音频线索列表
        text_cues: 文本线索列表（模型引用的字幕/文本内容）
        masked_modality: 被 mask 的模态 ("visual", "audio", "text", "none")
    
    Returns:
        幻觉统计结果
    """
    num_visual = len(visual_cues)
    num_audio = len(audio_cues)
    num_text = len(text_cues)
    
    # 全模态总线索数（视觉+听觉+文本）
    total_claims = num_visual + num_audio + num_text
    
    # 视觉+听觉总线索数（不包括文本）
    total_claims_vision_audio = num_visual + num_audio
    
    # 根据 mask 的模态计算幻觉数
    if masked_modality == "visual":
        num_hallucinated = num_visual  # mask视觉但生成了视觉线索 = 幻觉
    elif masked_modality == "audio":
        num_hallucinated = num_audio   # mask音频但生成了音频线索 = 幻觉
    elif masked_modality == "text":
        num_hallucinated = num_text    # mask文本但使用了文本 = 幻觉
    else:
        num_hallucinated = 0  # 没有 mask，不计算幻觉
    
    # 计算幻觉率（两个版本）
    # 1. 以全模态（V+A+T）为分母
    hallucination_rate = num_hallucinated / total_claims if total_claims > 0 else 0.0
    
    # 2. 以视觉+听觉（V+A）为分母（只在 mask 视觉或听觉时有意义）
    if masked_modality in ["visual", "audio"]:
        hallucination_rate_va = num_hallucinated / total_claims_vision_audio if total_claims_vision_audio > 0 else 0.0
    else:
        hallucination_rate_va = 0.0
    
    return {
        "num_visual_claims": num_visual,
        "num_audio_claims": num_audio,
        "num_text_claims": num_text,
        "total_claims": total_claims,
        "total_claims_vision_audio": total_claims_vision_audio,
        "num_hallucinated_claims": num_hallucinated,
        "hallucination_rate": hallucination_rate,
        "hallucination_rate_vision_audio": hallucination_rate_va,
        "masked_modality": masked_modality,
    }


def process_sample(sample_data: Dict, use_qwen: bool = False) -> Dict:
    """处理单个样本"""
    name = sample_data.get("name", "")
    think_content = sample_data.get("think", "")
    
    # 获取 counterfactual 配置
    metadata = sample_data.get("metadata", {})
    cf_config = metadata.get("counterfactual_config", {})
    masked_modality = cf_config.get("mask_modality", "none")  # "visual", "audio", "text", "none"
    
    # 调用 LLM 从 think 中提取线索（包括 visual, audio, text）
    if use_qwen:
        clues = call_qwen_extract_clues(think_content)
    else:
        clues = call_gpt_extract_clues(think_content)
    
    visual_cues = clues.get("visual_cues", [])
    audio_cues = clues.get("audio_cues", [])
    text_cues = clues.get("text_cues", [])
    
    # 计算幻觉指标
    metrics = compute_hallucination_metrics(
        visual_cues, audio_cues, text_cues, masked_modality
    )
    
    return {
        "name": name,
        "visual_cues": visual_cues,
        "audio_cues": audio_cues,
        "text_cues": text_cues,
        **metrics
    }


def process_worker(line_data: Tuple[int, str], use_qwen: bool) -> Optional[Dict]:
    """并发处理函数"""
    idx, line = line_data
    try:
        sample_data = json.loads(line)
        return process_sample(sample_data, use_qwen=use_qwen)
    except Exception as e:
        print(f"[ERROR] Processing line {idx} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_qwen(llm_name: str = "Qwen25", dataset_root: str = None):
    """初始化 Qwen 模型（参考 evaluation.py 的方式）"""
    global qwen_llm, qwen_tokenizer, qwen_sampling_params
    
    print(f"[INFO] Initializing Qwen model: {llm_name}")
    
    # 动态加载 AffectGPT 配置
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "affectgpt_local"))
    
    from affect_config import create_local_config
    from affectgpt_local import build_local_modules
    import types
    
    # 创建配置
    if dataset_root is None:
        dataset_root = os.environ.get("DATASET_ROOT", "/mnt/afs/hanzhiyuan/MER-UniBench/data")
    local_cfg = create_local_config(dataset_root)
    
    # 将配置转换为模块并注入
    cfg_module = types.ModuleType("config")
    for key, value in vars(local_cfg).items():
        setattr(cfg_module, key, value)
    sys.modules["config"] = cfg_module
    
    # 加载 LLM
    modules = build_local_modules()
    qwen_llm, qwen_tokenizer, qwen_sampling_params = modules["load_llm"](llm_name)
    
    print(f"[INFO] Qwen model loaded successfully")


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Hallucination Detection (CHAIR-M)")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径（counterfactual 推理结果）")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--engine",
        choices=["gpt", "qwen"],
        default="gpt",
        help="推理引擎：gpt 或 qwen（默认 gpt）"
    )
    parser.add_argument("--llm-name", default="Qwen25", help="Qwen 模型名称")
    parser.add_argument("--dataset-root", default=None, help="数据集根目录（用于 Qwen）")
    parser.add_argument("--max-samples", type=int, default=None, help="最大处理样本数（用于调试）")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="并发线程数")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    
    args = parser.parse_args()
    
    # 如果使用 Qwen，先初始化模型
    use_qwen = (args.engine == "qwen")
    if use_qwen:
        initialize_qwen(args.llm_name, args.dataset_root)
    
    print(f"[INFO] Loading input file: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 断点续跑
    processed_names = set()
    if args.resume and os.path.exists(args.output):
        print(f"[INFO] Resuming from existing output: {args.output}")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    name = obj.get("name")
                    if name:
                        processed_names.add(name)
                except:
                    pass
        print(f"[INFO] Found {len(processed_names)} processed samples")
    
    # 过滤待处理的行
    lines_to_process = []
    for idx, line in enumerate(lines[:args.max_samples] if args.max_samples else lines):
        try:
            data = json.loads(line)
            name = data.get("name", "")
            if args.resume and name in processed_names:
                continue
            lines_to_process.append((idx, line))
        except:
            pass
    
    print(f"[INFO] Processing {len(lines_to_process)} samples with engine={args.engine}")
    
    # Qwen (vLLM) 不是线程安全的，必须顺序处理；GPT 可以并发
    with open(args.output, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        if use_qwen:
            # 顺序处理 Qwen
            print("[INFO] Using sequential processing for Qwen (vLLM is not thread-safe)")
            for line_data in tqdm(lines_to_process, desc="Processing"):
                try:
                    result = process_worker(line_data, use_qwen)
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                except Exception as e:
                    print(f"[ERROR] Processing line {line_data[0]} failed: {e}")
        else:
            # 并发处理 GPT
            print(f"[INFO] Using concurrent processing with max_workers={args.max_workers}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_idx = {
                    executor.submit(process_worker, line_data, use_qwen): line_data[0]
                    for line_data in lines_to_process
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx)):
                    try:
                        result = future.result()
                        if result:
                            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                            f_out.flush()
                    except Exception as e:
                        print(f"[ERROR] Thread exception: {e}")
    
    print(f"[INFO] Done! Results saved to: {args.output}")
    
    # 读取结果并计算统计信息
    with open(args.output, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    if not results:
        print("No results to analyze.")
        return
    
    # 按 masked_modality 分组统计
    total_visual = sum(r["num_visual_claims"] for r in results)
    total_audio = sum(r["num_audio_claims"] for r in results)
    total_text = sum(r["num_text_claims"] for r in results)
    total_claims = sum(r["total_claims"] for r in results)
    total_claims_va = sum(r["total_claims_vision_audio"] for r in results)
    total_hallucinated = sum(r["num_hallucinated_claims"] for r in results)
    
    masked_modality = results[0].get("masked_modality", "unknown")
    
    # 计算平均线索数
    avg_visual = total_visual / len(results) if results else 0
    avg_audio = total_audio / len(results) if results else 0
    avg_text = total_text / len(results) if results else 0
    avg_total = total_claims / len(results) if results else 0
    avg_total_va = total_claims_va / len(results) if results else 0
    
    # 构建统计报告
    report_lines = []
    report_lines.append("=" * 80)
    
    # 根据模式选择不同的报告标题
    if masked_modality in ["none", "unknown", ""]:
        # Baseline 模式：不计算幻觉率，只统计线索
        report_lines.append("Multimodal Clue Extraction Statistics (Baseline Mode)")
        report_lines.append("=" * 80)
        report_lines.append(f"总样本数: {len(results)}")
        report_lines.append(f"模式: Baseline (无 Mask)")
        report_lines.append("")
        report_lines.append("线索统计 (Claims):")
        report_lines.append(f"  - 视觉线索总数 (N_V): {total_visual}")
        report_lines.append(f"  - 音频线索总数 (N_A): {total_audio}")
        report_lines.append(f"  - 文本线索总数 (N_T): {total_text}")
        report_lines.append(f"  - 全模态总计 (V+A+T): {total_claims}")
        report_lines.append(f"  - 视觉+听觉总计 (V+A): {total_claims_va}")
        report_lines.append("")
        report_lines.append("每样本平均线索数:")
        report_lines.append(f"  - 平均视觉线索: {avg_visual:.2f}")
        report_lines.append(f"  - 平均音频线索: {avg_audio:.2f}")
        report_lines.append(f"  - 平均文本线索: {avg_text:.2f}")
        report_lines.append(f"  - 平均全模态线索 (V+A+T): {avg_total:.2f}")
        report_lines.append(f"  - 平均视觉+听觉线索 (V+A): {avg_total_va:.2f}")
        report_lines.append("")
        report_lines.append("线索比例分布:")
        if total_claims > 0:
            report_lines.append(f"  - 视觉占比: {total_visual / total_claims * 100:.1f}%")
            report_lines.append(f"  - 音频占比: {total_audio / total_claims * 100:.1f}%")
            report_lines.append(f"  - 文本占比: {total_text / total_claims * 100:.1f}%")
        else:
            report_lines.append("  - (无线索)")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("说明：")
        report_lines.append("- 这是 Baseline 模式，没有对任何模态进行 Mask")
        report_lines.append("- 此统计可作为对照组，用于与 Mask 实验对比")
        report_lines.append("- 线索数量反映了模型对各模态信息的利用程度")
        report_lines.append("=" * 80)
    else:
        # Mask 模式：计算幻觉率
        avg_hallucination_rate = sum(r["hallucination_rate"] for r in results) / len(results)
        global_hallucination_rate = total_hallucinated / total_claims if total_claims > 0 else 0
        
        # 视觉+听觉幻觉率
        avg_hallucination_rate_va = sum(r["hallucination_rate_vision_audio"] for r in results) / len(results)
        global_hallucination_rate_va = total_hallucinated / total_claims_va if total_claims_va > 0 else 0
        
        report_lines.append("CHAIR-M Hallucination Statistics")
        report_lines.append("=" * 80)
        report_lines.append(f"总样本数: {len(results)}")
        report_lines.append(f"Masked Modality: {masked_modality}")
        report_lines.append("")
        report_lines.append("线索统计 (Claims):")
        report_lines.append(f"  - 视觉线索 (N_V): {total_visual}")
        report_lines.append(f"  - 音频线索 (N_A): {total_audio}")
        report_lines.append(f"  - 文本线索 (N_T): {total_text}")
        report_lines.append(f"  - 全模态总计 (V+A+T): {total_claims}")
        report_lines.append(f"  - 视觉+听觉总计 (V+A): {total_claims_va}")
        report_lines.append("")
        report_lines.append("每样本平均线索数:")
        report_lines.append(f"  - 平均视觉线索: {avg_visual:.2f}")
        report_lines.append(f"  - 平均音频线索: {avg_audio:.2f}")
        report_lines.append(f"  - 平均文本线索: {avg_text:.2f}")
        report_lines.append(f"  - 平均全模态线索 (V+A+T): {avg_total:.2f}")
        report_lines.append(f"  - 平均视觉+听觉线索 (V+A): {avg_total_va:.2f}")
        report_lines.append("")
        report_lines.append(f"幻觉统计 (在 mask_{masked_modality} 场景下):")
        report_lines.append(f"  - 幻觉线索数: {total_hallucinated}")
        report_lines.append("")
        report_lines.append("  [全模态 V+A+T 分母]")
        report_lines.append(f"    - 全局幻觉率: {global_hallucination_rate * 100:.2f}%")
        report_lines.append(f"    - 平均样本幻觉率: {avg_hallucination_rate * 100:.2f}%")
        report_lines.append("")
        if masked_modality in ["visual", "audio"]:
            report_lines.append("  [仅视觉+听觉 V+A 分母]")
            report_lines.append(f"    - 全局幻觉率: {global_hallucination_rate_va * 100:.2f}%")
            report_lines.append(f"    - 平均样本幻觉率: {avg_hallucination_rate_va * 100:.2f}%")
            report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("解释：")
        report_lines.append(f"- 在 mask_{masked_modality} 场景下，模型不应生成 {masked_modality} 相关线索")
        report_lines.append(f"- 但模型仍然生成了 {total_hallucinated} 个 {masked_modality} 线索")
        report_lines.append(f"- 全模态幻觉率 = {total_hallucinated} / {total_claims} = {global_hallucination_rate * 100:.2f}%")
        if masked_modality in ["visual", "audio"]:
            report_lines.append(f"- 视觉+听觉幻觉率 = {total_hallucinated} / {total_claims_va} = {global_hallucination_rate_va * 100:.2f}%")
        report_lines.append("=" * 80)
    
    # 打印到控制台
    print("\n" + "\n".join(report_lines))
    
    # 保存到文件（与输入文件同目录，文件名包含引擎类型）
    input_dir = os.path.dirname(args.input)
    engine_suffix = "Gpt" if args.engine == "gpt" else "Qwen"
    stats_file = os.path.join(input_dir, f"perception_hallucination_{engine_suffix}.txt")
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\n[INFO] Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
