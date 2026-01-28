#!/usr/bin/env python3
"""
重新处理 POPE 结果文件，从 <answer> 标签中提取答案并重新判断 is_correct

规则：
1. 从 full_response 中提取 <answer>...</answer> 中的内容
2. 如果提取出 "yes"（不论大小写），且 expected_answer 是 "no"，则 is_correct = False
3. 如果没有提取出 "yes" 或 "no"，也算错 is_correct = False
4. 只有答案与 expected_answer 匹配才 is_correct = True

使用示例：
python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/reprocess_pope_results.py \
    --input /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_a253.jsonl \
    --output /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_a253_fixed.jsonl
    
python /mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/merbench/reprocess_pope_results.py \
    --input /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_all_v253.jsonl \
    --output /mnt/afs/hanzhiyuan/huggingface/humanomniv2/pope_results_all_v253_fixed.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def extract_answer_from_tag(full_response: str) -> str:
    """
    从 full_response 中提取 <answer>...</answer> 标签中的内容
    
    Returns:
        提取的答案（小写），如果没找到则返回 "unknown"
    """
    # 匹配 <answer>...</answer> 标签（不区分大小写）
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, full_response, re.IGNORECASE | re.DOTALL)
    
    if match:
        answer = match.group(1).strip().lower()
        # 清理答案，只保留 yes 或 no
        if 'yes' in answer:
            return 'yes'
        elif 'no' in answer:
            return 'no'
        else:
            return answer[:50]  # 返回前50字符用于调试
    
    return "unknown"


def reprocess_pope_results(input_path: str, output_path: str, verbose: bool = False):
    """重新处理 POPE 结果文件"""
    
    results = []
    stats = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'unknown': 0,
        'changed': 0,
    }
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            stats['total'] += 1
            
            # 提取 <answer> 标签中的内容
            extracted_answer = extract_answer_from_tag(item.get('full_response', ''))
            expected_answer = item.get('expected_answer', 'no').lower()
            old_is_correct = item.get('is_correct', False)
            
            # 判断 is_correct
            if extracted_answer == 'unknown':
                # 没有提取出有效答案，算错
                is_correct = False
                stats['unknown'] += 1
            elif extracted_answer == expected_answer:
                # 答案匹配
                is_correct = True
                stats['correct'] += 1
            else:
                # 答案不匹配
                is_correct = False
                stats['incorrect'] += 1
            
            # 检查是否有变化
            if is_correct != old_is_correct:
                stats['changed'] += 1
                if verbose:
                    print(f"[CHANGED] {item['name']}: {old_is_correct} -> {is_correct}")
                    print(f"  extracted: '{extracted_answer}', expected: '{expected_answer}'")
            
            # 更新结果
            item['model_answer'] = extracted_answer
            item['is_correct'] = is_correct
            results.append(item)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 打印统计
    print(f"\n统计结果:")
    print(f"  总样本数: {stats['total']}")
    print(f"  正确数: {stats['correct']}")
    print(f"  错误数: {stats['incorrect']}")
    print(f"  无法提取答案: {stats['unknown']}")
    print(f"  变化数: {stats['changed']}")
    if stats['total'] > 0:
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"  准确率: {accuracy:.2f}%")
    
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="重新处理 POPE 结果文件")
    parser.add_argument("--input", type=str, required=True,
                        help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSONL 文件路径（默认在原文件名后加 _fixed）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细变化信息")
    parser.add_argument("--inplace", action="store_true",
                        help="原地修改文件")
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.inplace:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {output_path}")
    
    reprocess_pope_results(args.input, output_path, args.verbose)


if __name__ == "__main__":
    main()

