"""
情绪轮词汇分析脚本

分析情绪轮三层结构中词汇的 token 数量分布：
1. format.csv - 第一层同义词/形态变化
2. synonym.xlsx - 第二层近义词
3. wheel1-5.xlsx - 第三层情绪轮层级

使用方法:
    /mnt/afs/hanzhiyuan/.conda/envs/humanomni_v2/bin/python analyze_emotion_vocab.py
"""

import pandas as pd
import os
import sys
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_wheel_root():
    """获取情绪轮数据根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "emotion_wheel")


def collect_wheel_words(wheel_root: str) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Dict]]:
    """
    收集 wheel1-5.xlsx 中的所有词
    
    Returns:
        all_words: 所有词的集合
        wheel_words_by_level: {"level1": set, "level2": set, "level3": set}
        wheel_structure: 情绪轮完整结构 {wheel_name: {level1: {level2: [level3s]}}}
    """
    all_words = set()
    wheel_words_by_level = {"level1": set(), "level2": set(), "level3": set()}
    wheel_structure = {}
    
    for i in range(1, 6):
        xlsx_path = os.path.join(wheel_root, f"wheel{i}.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"Warning: {xlsx_path} not found")
            continue
            
        df = pd.read_excel(xlsx_path)
        wheel_name = f"wheel{i}"
        wheel_structure[wheel_name] = {}
        
        level1 = level2 = ""
        for _, row in df.iterrows():
            if not pd.isna(row["level1"]):
                level1 = str(row["level1"]).lower().strip()
                wheel_words_by_level["level1"].add(level1)
                all_words.add(level1)
                wheel_structure[wheel_name][level1] = {}
            
            if not pd.isna(row["level2"]):
                level2 = str(row["level2"]).lower().strip()
                wheel_words_by_level["level2"].add(level2)
                all_words.add(level2)
                if level1 and level1 in wheel_structure[wheel_name]:
                    wheel_structure[wheel_name][level1][level2] = []
            
            if not pd.isna(row["level3"]):
                level3 = str(row["level3"]).lower().strip()
                wheel_words_by_level["level3"].add(level3)
                all_words.add(level3)
                if level1 and level2:
                    if level1 in wheel_structure[wheel_name]:
                        if level2 in wheel_structure[wheel_name][level1]:
                            wheel_structure[wheel_name][level1][level2].append(level3)
    
    return all_words, wheel_words_by_level, wheel_structure


def collect_synonym_words(wheel_root: str) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    收集 synonym.xlsx 中的所有词
    
    Returns:
        all_words: 所有词的集合
        synonym_mapping: {word: [synonyms]}
    """
    xlsx_path = os.path.join(wheel_root, "synonym.xlsx")
    if not os.path.exists(xlsx_path):
        print(f"Warning: {xlsx_path} not found")
        return set(), {}
    
    df = pd.read_excel(xlsx_path)
    
    all_words = set()
    synonym_mapping = defaultdict(set)
    
    for run in range(1, 9):
        word_col = f"word_run{run}"
        syn_col = f"synonym_run{run}"
        
        for _, row in df.iterrows():
            if pd.isna(row[word_col]):
                continue
                
            word = str(row[word_col]).lower().strip()
            all_words.add(word)
            
            if not pd.isna(row[syn_col]):
                syns = str(row[syn_col]).split(",")
                for s in syns:
                    s = s.strip().lower()
                    if s:
                        all_words.add(s)
                        synonym_mapping[word].add(s)
                        # 反向映射：近义词也指向原词
                        synonym_mapping[s].add(word)
    
    # 转换为 list
    synonym_mapping = {k: list(v) for k, v in synonym_mapping.items()}
    
    return all_words, synonym_mapping


def collect_format_words(wheel_root: str) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    收集 format.csv 中的所有词
    
    Returns:
        all_words: 所有词的集合
        format_mapping: {word: [format_variants]}
    """
    csv_path = os.path.join(wheel_root, "format.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return set(), {}
    
    df = pd.read_csv(csv_path)
    
    all_words = set()
    format_mapping = defaultdict(set)
    
    for _, row in df.iterrows():
        name = str(row["name"]).lower().strip()
        all_words.add(name)
        format_mapping[name].add(name)
        
        if pd.notna(row["format"]):
            formats = str(row["format"]).split(",")
            for f in formats:
                f = f.strip().lower()
                if f:
                    all_words.add(f)
                    format_mapping[name].add(f)
                    # 反向映射：变体也指向规范名
                    format_mapping[f].add(name)
    
    # 转换为 list
    format_mapping = {k: list(v) for k, v in format_mapping.items()}
    
    return all_words, format_mapping


def analyze_token_counts(words: Set[str], name: str, tokenizer) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    分析词汇的 token 数量分布
    
    Returns:
        single_token: 单 token 词列表
        multi_token: (word, token_count) 列表
    """
    single_token = []
    multi_token = []
    
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1:
            single_token.append(word)
        else:
            multi_token.append((word, len(encoded)))
    
    print(f"\n=== {name} ===")
    print(f"总词数: {len(words)}")
    print(f"单 token 词: {len(single_token)} ({100*len(single_token)/len(words):.1f}%)")
    print(f"多 token 词: {len(multi_token)} ({100*len(multi_token)/len(words):.1f}%)")
    
    return single_token, multi_token


def build_complete_mapping(wheel_root: str):
    """
    构建完整的三层映射关系
    
    映射方向:
    输入词 -> format (形态归一化) -> synonym (近义词) -> wheel (情绪轮层级)
    """
    # 收集所有数据
    format_words, format_mapping = collect_format_words(wheel_root)
    synonym_words, synonym_mapping = collect_synonym_words(wheel_root)
    wheel_words, wheel_by_level, wheel_structure = collect_wheel_words(wheel_root)
    
    # 构建情绪轮反向索引: word -> (wheel_name, level1, level2, level3)
    wheel_reverse_index = {}
    for wheel_name, levels in wheel_structure.items():
        for level1, level2_dict in levels.items():
            wheel_reverse_index[level1] = (wheel_name, level1, None, None)
            for level2, level3_list in level2_dict.items():
                wheel_reverse_index[level2] = (wheel_name, level1, level2, None)
                for level3 in level3_list:
                    wheel_reverse_index[level3] = (wheel_name, level1, level2, level3)
    
    return {
        "format_mapping": format_mapping,
        "synonym_mapping": synonym_mapping,
        "wheel_structure": wheel_structure,
        "wheel_reverse_index": wheel_reverse_index,
        "wheel_by_level": wheel_by_level,
        "all_words": {
            "format": format_words,
            "synonym": synonym_words,
            "wheel": wheel_words,
        }
    }


def main():
    from transformers import AutoTokenizer
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
        local_files_only=True,
        trust_remote_code=True
    )
    
    wheel_root = get_wheel_root()
    print(f"情绪轮数据目录: {wheel_root}")
    
    print("\n" + "="*60)
    print("情绪轮词汇 Token 分析")
    print("="*60)
    
    # 1. 分析情绪轮核心词（wheel1-5.xlsx）
    wheel_words, wheel_by_level, wheel_structure = collect_wheel_words(wheel_root)
    print(f"\n【情绪轮层级分布】")
    print(f"  Level1 (最内层/核心情绪): {len(wheel_by_level['level1'])} 个")
    print(f"  Level2 (中间层): {len(wheel_by_level['level2'])} 个")
    print(f"  Level3 (最外层): {len(wheel_by_level['level3'])} 个")
    print(f"  去重后总数: {len(wheel_words)} 个")
    
    single_wheel, multi_wheel = analyze_token_counts(wheel_words, "情绪轮词汇 (wheel1-5.xlsx)", tokenizer)
    
    # 打印单 token 词
    print(f"\n单 token 情绪轮词 (全部 {len(single_wheel)} 个):")
    for w in sorted(single_wheel):
        tid = tokenizer.encode(w, add_special_tokens=False)[0]
        # 判断属于哪个层级
        levels = []
        if w in wheel_by_level["level1"]:
            levels.append("L1")
        if w in wheel_by_level["level2"]:
            levels.append("L2")
        if w in wheel_by_level["level3"]:
            levels.append("L3")
        print(f"  '{w}' -> token_id={tid}, levels={levels}")
    
    # 打印多 token 词示例
    print(f"\n多 token 情绪轮词 (共 {len(multi_wheel)} 个, 按 token 数降序):")
    for w, n in sorted(multi_wheel, key=lambda x: x[1], reverse=True):
        tids = tokenizer.encode(w, add_special_tokens=False)
        print(f"  '{w}' -> {n} tokens: {tids}")
    
    # 2. 分析近义词（synonym.xlsx）
    synonym_words, synonym_mapping = collect_synonym_words(wheel_root)
    single_syn, multi_syn = analyze_token_counts(synonym_words, "近义词 (synonym.xlsx)", tokenizer)
    
    # 3. 分析形态变化（format.csv）
    format_words, format_mapping = collect_format_words(wheel_root)
    single_fmt, multi_fmt = analyze_token_counts(format_words, "形态变化 (format.csv)", tokenizer)
    
    # 4. 汇总
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    all_words = wheel_words | synonym_words | format_words
    print(f"三层合并后总词数: {len(all_words)}")
    
    single_all, multi_all = analyze_token_counts(all_words, "全部词汇", tokenizer)
    
    # 5. 分析情绪轮结构
    print("\n" + "="*60)
    print("情绪轮结构详情")
    print("="*60)
    for wheel_name, levels in wheel_structure.items():
        print(f"\n【{wheel_name}】")
        for level1, level2_dict in levels.items():
            print(f"  {level1}:")
            for level2, level3_list in level2_dict.items():
                print(f"    {level2}: {level3_list}")


if __name__ == "__main__":
    main()
