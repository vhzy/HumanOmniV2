"""
情绪轮详细统计分析脚本

功能：
- 统计每个情绪轮内部各 Level1 类别的词数和 Token 数
- 展示三层映射的展开过程
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from emotion_token_map import EmotionTokenMapBuilder


def analyze_wheel_stats(tokenizer_path: str = "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B"):
    """分析每个情绪轮的详细统计"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    builder = EmotionTokenMapBuilder(tokenizer=tokenizer, mapping_depth="full")
    builder.build()
    
    print("=" * 80)
    print("每个情绪轮内部的详细统计")
    print("=" * 80)
    
    for wheel_name in sorted(builder.wheel_structure.keys()):
        wheel_data = builder.wheel_structure[wheel_name]  # {l1: {l2: [l3_words]}}
        wheel_conflicts = builder.per_wheel_conflicts.get(wheel_name, set())
        
        print(f"\n{'=' * 80}")
        print(f"【{wheel_name.upper()}】 Level1 类别: {len(wheel_data)}, 冲突 Token: {len(wheel_conflicts)}")
        print(f"{'=' * 80}")
        
        wheel_total_words = 0
        wheel_total_raw = 0
        wheel_total_safe = 0
        
        for l1 in sorted(wheel_data.keys()):
            l2_data = wheel_data[l1]  # {l2: [l3_words]}
            
            # 收集该 L1 下所有词（wheel 结构中的词）
            l1_all_words = set()
            l1_all_words.add(l1)  # L1 本身
            
            for l2, l3_words in l2_data.items():
                l1_all_words.add(l2)  # L2 词
                l1_all_words.update(l3_words)  # L3 词
            
            # 获取这些词的 tokens（经过三层映射展开后）
            l1_raw_tokens = set()
            for word in l1_all_words:
                if word in builder.category_to_raw_tokens:
                    l1_raw_tokens.update(builder.category_to_raw_tokens[word])
            
            l1_safe_tokens = l1_raw_tokens - wheel_conflicts
            rate = len(l1_safe_tokens) / len(l1_raw_tokens) * 100 if l1_raw_tokens else 0
            
            print(f"\n  【{l1}】{len(l1_all_words)} 词, {len(l1_raw_tokens)} → {len(l1_safe_tokens)} tokens ({rate:.1f}%)")
            
            # 展示 Level2 详情
            for l2 in sorted(l2_data.keys()):
                l3_words = l2_data[l2]
                l2_all_words = set([l2]) | set(l3_words)
                l2_raw = set()
                for w in l2_all_words:
                    if w in builder.category_to_raw_tokens:
                        l2_raw.update(builder.category_to_raw_tokens[w])
                l2_safe = l2_raw - wheel_conflicts
                print(f"      └─ {l2}: {len(l2_all_words)}词, {len(l2_raw)}→{len(l2_safe)} tokens")
            
            wheel_total_words += len(l1_all_words)
            wheel_total_raw += len(l1_raw_tokens)
            wheel_total_safe += len(l1_safe_tokens)
        
        rate = wheel_total_safe / wheel_total_raw * 100 if wheel_total_raw else 0
        print(f"\n  ▸ {wheel_name} 总计: {wheel_total_words} 词, {wheel_total_raw} → {wheel_total_safe} tokens ({rate:.1f}%)")
    
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)


def explain_mapping_example(tokenizer_path: str = "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B"):
    """
    详细解释一个具体例子的映射过程
    
    以 wheel2 的 sad 类别为例：
    - 19词：wheel 结构中的原始词条数
    - 647 tokens：经过三层映射展开后的首 token 总数
    - 431 tokens：移除 wheel 内冲突后的有效 token 数
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    builder = EmotionTokenMapBuilder(tokenizer=tokenizer, mapping_depth="full")
    builder.build()
    
    print("=" * 80)
    print("详细解释：以 wheel2 的 【sad】 类别为例")
    print("=" * 80)
    
    wheel_name = "wheel2"
    l1 = "sad"
    wheel_data = builder.wheel_structure[wheel_name]
    wheel_conflicts = builder.per_wheel_conflicts.get(wheel_name, set())
    l2_data = wheel_data[l1]
    
    # Step 1: Wheel 结构中的原始词
    print("\n【第一层】Wheel 结构中的词条（19词）")
    print("-" * 60)
    
    wheel_words = set()
    wheel_words.add(l1)
    print(f"  Level1: {l1}")
    
    for l2, l3_words in sorted(l2_data.items()):
        wheel_words.add(l2)
        wheel_words.update(l3_words)
        print(f"  Level2: {l2} -> Level3: {l3_words}")
    
    print(f"\n  共计: {len(wheel_words)} 个独立词条")
    print(f"  词条列表: {sorted(wheel_words)}")
    
    # Step 2: 三层映射展开
    print("\n" + "=" * 80)
    print("【第二层】三层映射展开过程")
    print("-" * 60)
    
    total_expanded_words = set()
    for word in sorted(wheel_words)[:5]:  # 只展示前5个
        expanded = builder.category_to_words.get(word, set())
        total_expanded_words.update(expanded)
        print(f"\n  '{word}' 展开为 {len(expanded)} 个同义词:")
        for w in sorted(expanded)[:10]:
            print(f"    - {w}")
        if len(expanded) > 10:
            print(f"    ... 共 {len(expanded)} 个")
    
    # 获取全部展开词
    all_expanded = set()
    for word in wheel_words:
        all_expanded.update(builder.category_to_words.get(word, set()))
    
    print(f"\n  19 个原始词条 -> {len(all_expanded)} 个展开词")
    
    # Step 3: Token 提取
    print("\n" + "=" * 80)
    print("【第三层】Token 提取（6种变体）")
    print("-" * 60)
    
    sample_word = "sad"
    print(f"\n  以 '{sample_word}' 为例，提取 6 种变体的首 Token:")
    variants = [sample_word, f" {sample_word}"]
                # sample_word.capitalize(), f" {sample_word.capitalize()}",
                # sample_word.upper(), f" {sample_word.upper()}"]
    
    for v in variants:
        tokens = tokenizer.encode(v, add_special_tokens=False)
        first_token = tokens[0] if tokens else None
        decoded = tokenizer.decode([first_token]) if first_token else ""
        print(f"    '{v}' -> Token ID: {first_token}, 解码: '{decoded}'")
    
    # Step 4: 统计
    print("\n" + "=" * 80)
    print("【统计汇总】")
    print("-" * 60)
    
    l1_raw_tokens = set()
    for word in wheel_words:
        if word in builder.category_to_raw_tokens:
            l1_raw_tokens.update(builder.category_to_raw_tokens[word])
    
    l1_safe_tokens = l1_raw_tokens - wheel_conflicts
    
    print(f"""
  原始词条数:     {len(wheel_words)} 词（wheel 结构中的 L1+L2+L3）
  展开词数:       {len(all_expanded)} 词（经过 format.csv + synonym.xlsx 展开）
  原始 Token 数:  {len(l1_raw_tokens)}（每个展开词的 6 种变体首 Token）
  冲突 Token 数:  {len(l1_raw_tokens - l1_safe_tokens)}（在该 wheel 内多个 L1 类别共享）
  有效 Token 数:  {len(l1_safe_tokens)}（移除冲突后）
  保留率:         {len(l1_safe_tokens) / len(l1_raw_tokens) * 100:.1f}%
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="情绪轮统计分析")
    parser.add_argument("--explain", action="store_true", help="展示详细映射解释")
    parser.add_argument("--tokenizer", default="/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B", help="Tokenizer 路径")
    
    args = parser.parse_args()
    
    if args.explain:
        explain_mapping_example(args.tokenizer)
    else:
        analyze_wheel_stats(args.tokenizer)
