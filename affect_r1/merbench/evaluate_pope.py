#!/usr/bin/env python3
"""
POPE压力测试评估脚本

功能：
1. 读取POPE推理结果
2. 计算幻觉率（Spurious Hallucination Rate）
3. 生成详细统计报告

指标定义：
- Accuracy: 回答正确的比例（回答"no"是正确的，因为模态被mask了）
- Hallucination Rate: 错误回答"yes"的比例（模型产生了虚假关联幻觉）
- Precision/Recall/F1: 基于二分类的标准指标

使用示例：
python -m affect_r1.merbench.evaluate_pope \
    --input /path/to/pope_results.jsonl \
    --output /path/to/pope_evaluation.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_pope_results(results: List[Dict]) -> Dict:
    """
    评估POPE结果
    
    预期答案是 "no"（因为模态被mask了）
    回答 "yes" 意味着模型产生了幻觉
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}
    
    # 统计
    correct = 0
    incorrect = 0
    yes_answers = 0
    no_answers = 0
    other_answers = 0
    
    # 按样本分组统计
    sample_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for r in results:
        name = r.get("name", "unknown")
        expected = r.get("expected_answer", "no")
        model_answer = r.get("model_answer", "").lower().strip()
        is_correct = r.get("is_correct", model_answer == expected)
        
        sample_stats[name]["total"] += 1
        
        if is_correct:
            correct += 1
            sample_stats[name]["correct"] += 1
        else:
            incorrect += 1
        
        if model_answer == "yes":
            yes_answers += 1
        elif model_answer == "no":
            no_answers += 1
        else:
            other_answers += 1
    
    # 计算指标
    accuracy = correct / total
    hallucination_rate = yes_answers / total  # 回答yes就是幻觉
    
    # 二分类指标（预测no=正确，预测yes=错误）
    # True Positive: 预测no，实际应该是no
    # False Positive: 预测yes，实际应该是no（幻觉）
    tp = no_answers  # 正确回答no
    fp = yes_answers  # 错误回答yes（幻觉）
    fn = 0  # 在POPE测试中没有FN（因为所有正确答案都是no）
    tn = 0  # 在POPE测试中没有TN
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 1.0  # 因为所有正确答案都是no，recall=1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 样本级别统计
    sample_level_accuracy = sum(
        1 for s in sample_stats.values() if s["correct"] == s["total"]
    ) / len(sample_stats) if sample_stats else 0
    
    return {
        "total_questions": total,
        "total_samples": len(sample_stats),
        "correct_answers": correct,
        "incorrect_answers": incorrect,
        "yes_answers": yes_answers,
        "no_answers": no_answers,
        "other_answers": other_answers,
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sample_level_accuracy": sample_level_accuracy,
    }


def generate_report(metrics: Dict, mask_type: str) -> str:
    """生成详细报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("POPE压力测试评估报告 (Spurious Hallucination Evaluation)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Mask类型: {mask_type}")
    lines.append(f"总问题数: {metrics['total_questions']}")
    lines.append(f"总样本数: {metrics['total_samples']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("答案分布")
    lines.append("-" * 70)
    lines.append(f"  'no' 回答数:    {metrics['no_answers']} ({metrics['no_answers']/metrics['total_questions']*100:.1f}%)")
    lines.append(f"  'yes' 回答数:   {metrics['yes_answers']} ({metrics['yes_answers']/metrics['total_questions']*100:.1f}%)")
    lines.append(f"  其他回答数:     {metrics['other_answers']} ({metrics['other_answers']/metrics['total_questions']*100:.1f}%)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("核心指标")
    lines.append("-" * 70)
    lines.append(f"  准确率 (Accuracy):        {metrics['accuracy']*100:.2f}%")
    lines.append(f"  幻觉率 (Hallucination):   {metrics['hallucination_rate']*100:.2f}%")
    lines.append(f"  样本级准确率:             {metrics['sample_level_accuracy']*100:.2f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("二分类指标")
    lines.append("-" * 70)
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall:    {metrics['recall']:.4f}")
    lines.append(f"  F1 Score:  {metrics['f1']:.4f}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("解释")
    lines.append("=" * 70)
    if mask_type == "audio":
        lines.append("场景：音频被静音，问模型是否听到某个音频线索")
        lines.append("正确答案应该是 'no'（因为音频被mask了）")
        lines.append("如果模型回答 'yes'，说明模型产生了视听共现幻觉")
        lines.append("  - 模型根据视频内容推断出了音频信息（错误）")
    elif mask_type == "visual":
        lines.append("场景：视频被黑屏，问模型是否看到某个视觉线索")
        lines.append("正确答案应该是 'no'（因为视频被mask了）")
        lines.append("如果模型回答 'yes'，说明模型产生了视听共现幻觉")
        lines.append("  - 模型根据音频内容推断出了视觉信息（错误）")
    lines.append("")
    lines.append(f"幻觉率 {metrics['hallucination_rate']*100:.2f}% 表示模型在 {metrics['yes_answers']} 个问题上")
    lines.append("产生了虚假关联幻觉（错误地认为感知到了被mask的内容）")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="POPE压力测试评估")
    parser.add_argument("--input", type=str, required=True,
                        help="POPE推理结果文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果输出路径（默认在input同目录）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POPE压力测试评估")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print()
    
    # 加载结果
    print("[1/2] 加载推理结果...")
    results = load_jsonl(args.input)
    print(f"  加载样本数: {len(results)}")
    
    if not results:
        print("[ERROR] 没有结果可评估")
        return
    
    # 获取mask类型
    mask_type = results[0].get("mask_type", "unknown")
    
    # 评估
    print("\n[2/2] 计算评估指标...")
    metrics = evaluate_pope_results(results)
    
    # 生成报告
    report = generate_report(metrics, mask_type)
    print("\n" + report)
    
    # 保存结果
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"pope_evaluation_{mask_type}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "mask_type": mask_type,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存: {output_path}")
    
    # 保存文本报告
    report_path = output_path.with_suffix('.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"文本报告已保存: {report_path}")


if __name__ == "__main__":
    main()

