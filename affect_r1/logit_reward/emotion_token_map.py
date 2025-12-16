"""
情感 Token 映射表构建模块

Phase 0: 离线预处理 (Offline Preparation)

功能：
1. 词库展开 - 从情感轮三层结构中收集所有同义词
2. 首 Token 提取与变体覆盖 - 覆盖无空格、有空格、首字母大写等变体
3. 互斥性清洗 - 移除跨类别歧义的 Token ID
4. 构建静态映射表 Clean_Vocab_Map: {category: Set[Safe_Token_IDs]}
"""

import os
import json
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import torch


@dataclass
class TokenMapStats:
    """Token 映射统计信息"""
    category: str
    words_before_filter: int
    tokens_before_filter: int
    tokens_after_filter: int
    removed_tokens: int
    removed_token_examples: List[str] = field(default_factory=list)


class EmotionTokenMapBuilder:
    """
    情感 Token 映射表构建器
    
    按照 Phase 0 流程构建无歧义的 Token ID 映射表
    """
    
    def __init__(
        self,
        tokenizer,
        emotion_wheel_root: Optional[str] = None,
        mapping_depth: str = "full",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            emotion_wheel_root: 情绪轮数据根目录
            mapping_depth: 映射深度 ("wheel_only", "wheel_synonym", "full")
        """
        self.tokenizer = tokenizer
        self.mapping_depth = mapping_depth
        
        if emotion_wheel_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            emotion_wheel_root = os.path.join(current_dir, "..", "emotion_wheel")
        self.emotion_wheel_root = emotion_wheel_root
        
        # 三层数据结构
        self.format_to_canonical: Dict[str, str] = {}
        self.canonical_to_formats: Dict[str, Set[str]] = defaultdict(set)
        self.word_to_synonyms: Dict[str, Set[str]] = defaultdict(set)
        self.wheel_structure: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        self.wheel_reverse_index: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}
        self.wheel_words_by_level: Dict[str, Set[str]] = {
            "level1": set(), "level2": set(), "level3": set()
        }
        
        # Phase 0 结果
        self.category_to_words: Dict[str, Set[str]] = {}
        self.category_to_raw_tokens: Dict[str, Set[int]] = {}
        self.category_to_safe_tokens: Dict[str, Set[int]] = {}
        self.global_token_count: Dict[int, int] = defaultdict(int)
        self.conflicting_tokens: Set[int] = set()
        
        # 统计信息
        self.stats: List[TokenMapStats] = []
        
        # 加载数据
        self._load_all_data()
    
    # ================================================================
    # 数据加载
    # ================================================================
    
    def _load_format_csv(self):
        csv_path = os.path.join(self.emotion_wheel_root, "format.csv")
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            name = str(row["name"]).lower().strip()
            self.format_to_canonical[name] = name
            self.canonical_to_formats[name].add(name)
            
            if pd.notna(row["format"]):
                for f in str(row["format"]).split(","):
                    f = f.strip().lower()
                    if f:
                        self.format_to_canonical[f] = name
                        self.canonical_to_formats[name].add(f)
    
    def _load_synonym_xlsx(self):
        xlsx_path = os.path.join(self.emotion_wheel_root, "synonym.xlsx")
        if not os.path.exists(xlsx_path):
            return
        
        df = pd.read_excel(xlsx_path)
        for run in range(1, 9):
            word_col = f"word_run{run}"
            syn_col = f"synonym_run{run}"
            
            for _, row in df.iterrows():
                if pd.isna(row[word_col]):
                    continue
                
                word = str(row[word_col]).lower().strip()
                self.word_to_synonyms[word].add(word)
                
                if not pd.isna(row[syn_col]):
                    for s in str(row[syn_col]).split(","):
                        s = s.strip().lower()
                        if s:
                            self.word_to_synonyms[word].add(s)
                            self.word_to_synonyms[s].add(word)
    
    def _load_wheel_xlsx(self):
        for i in range(1, 6):
            xlsx_path = os.path.join(self.emotion_wheel_root, f"wheel{i}.xlsx")
            if not os.path.exists(xlsx_path):
                continue
            
            df = pd.read_excel(xlsx_path)
            wheel_name = f"wheel{i}"
            self.wheel_structure[wheel_name] = {}
            
            level1 = level2 = ""
            for _, row in df.iterrows():
                if not pd.isna(row["level1"]):
                    level1 = str(row["level1"]).lower().strip()
                    self.wheel_words_by_level["level1"].add(level1)
                    self.wheel_structure[wheel_name][level1] = {}
                    self.wheel_reverse_index[level1] = (wheel_name, level1, None, None)
                
                if not pd.isna(row["level2"]):
                    level2 = str(row["level2"]).lower().strip()
                    self.wheel_words_by_level["level2"].add(level2)
                    if level1:
                        self.wheel_structure[wheel_name][level1][level2] = []
                    self.wheel_reverse_index[level2] = (wheel_name, level1, level2, None)
                
                if not pd.isna(row["level3"]):
                    level3 = str(row["level3"]).lower().strip()
                    self.wheel_words_by_level["level3"].add(level3)
                    if level1 and level2:
                        self.wheel_structure[wheel_name][level1][level2].append(level3)
                    self.wheel_reverse_index[level3] = (wheel_name, level1, level2, level3)
    
    def _load_all_data(self):
        self._load_format_csv()
        self._load_synonym_xlsx()
        self._load_wheel_xlsx()
    
    # ================================================================
    # Phase 0 Step 1: 词库展开
    # ================================================================
    
    def _expand_category_to_words(self, category: str) -> Set[str]:
        """将情绪轮类别扩展为所有相关词汇"""
        category = category.lower().strip()
        
        if category not in self.wheel_reverse_index:
            return set()
        
        wheel_name, l1, l2, l3 = self.wheel_reverse_index[category]
        words = set()
        
        if wheel_name in self.wheel_structure:
            wheel = self.wheel_structure[wheel_name]
            
            if l1 == category and l2 is None:
                words.add(l1)
                if l1 in wheel:
                    for l2_word, l3_list in wheel[l1].items():
                        words.add(l2_word)
                        words.update(l3_list)
            elif l2 == category and l3 is None:
                words.add(l2)
                if l1 in wheel and l2 in wheel[l1]:
                    words.update(wheel[l1][l2])
            elif l3 == category:
                words.add(l3)
        
        if self.mapping_depth == "wheel_only":
            return words
        
        if self.mapping_depth in ["wheel_synonym", "full"]:
            expanded = set()
            for w in words:
                expanded.add(w)
                if w in self.word_to_synonyms:
                    expanded.update(self.word_to_synonyms[w])
            words = expanded
        
        if self.mapping_depth == "full":
            expanded = set()
            for w in words:
                expanded.add(w)
                if w in self.canonical_to_formats:
                    expanded.update(self.canonical_to_formats[w])
            words = expanded
        
        return words
    
    # ================================================================
    # Phase 0 Step 2: 首 Token 提取与变体覆盖
    # ================================================================
    
    def _get_first_token_variants(self, word: str) -> Set[int]:
        """
        获取一个词所有可能的首 Token ID
        
        覆盖变体：无空格、有空格、首字母大写等
        """
        token_ids = set()
        
        variants = [
            word,
            " " + word,
            # word.capitalize(),
            # " " + word.capitalize(),
            # word.upper(),
            # " " + word.upper(),
        ]
        
        for variant in variants:
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                token_ids.add(encoded[0])
        
        return token_ids
    
    # ================================================================
    # Phase 0 Step 3: 互斥性清洗 (每个情绪轮内部单独检测)
    # ================================================================
    
    def _get_wheel_and_level1_for_category(self, category: str) -> Optional[Tuple[str, str]]:
        """获取类别对应的 (wheel_name, level1)"""
        if category not in self.wheel_reverse_index:
            return None
        wheel_name, l1, _, _ = self.wheel_reverse_index[category]
        return (wheel_name, l1)
    
    def _build_per_wheel_level1_token_sets(self) -> Dict[str, Dict[str, Set[int]]]:
        """
        构建每个情绪轮内部的 Level1 Token 集合
        
        Returns:
            {wheel_name: {level1_category: set(token_ids)}}
        """
        wheel_level1_tokens: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        
        for category, tokens in self.category_to_raw_tokens.items():
            result = self._get_wheel_and_level1_for_category(category)
            if result:
                wheel_name, l1 = result
                wheel_level1_tokens[wheel_name][l1].update(tokens)
        
        return wheel_level1_tokens
    
    def _compute_per_wheel_conflicts(self) -> Dict[str, Set[int]]:
        """
        每个情绪轮内部单独检测冲突
        
        Returns:
            {wheel_name: set(conflicting_token_ids)}
        """
        wheel_level1_tokens = self._build_per_wheel_level1_token_sets()
        wheel_conflicts: Dict[str, Set[int]] = {}
        
        for wheel_name, level1_tokens in wheel_level1_tokens.items():
            # 统计每个 token 在该 wheel 内出现在多少个 Level1 类别中
            token_count: Dict[int, int] = defaultdict(int)
            for l1_cat, tokens in level1_tokens.items():
                for tid in tokens:
                    token_count[tid] += 1
            
            # 标记冲突 token（出现在 >1 个 Level1 类别中）
            conflicts = {tid for tid, count in token_count.items() if count > 1}
            wheel_conflicts[wheel_name] = conflicts
            
            print(f"  {wheel_name}: {len(level1_tokens)} 个 Level1 类别, {len(conflicts)} 个冲突 Token")
        
        return wheel_conflicts
    
    def _compute_global_token_counts(self):
        """计算每个情绪轮内部的冲突"""
        self.per_wheel_conflicts = self._compute_per_wheel_conflicts()
        
        # 合并所有 wheel 的冲突 token（用于统计）
        self.conflicting_tokens = set()
        for conflicts in self.per_wheel_conflicts.values():
            self.conflicting_tokens.update(conflicts)
    
    def _identify_conflicting_tokens(self) -> Set[int]:
        return self.conflicting_tokens
    
    def _clean_tokens(self):
        """
        清洗 token：每个类别只移除其所属 wheel 内的冲突 token
        """
        for category, raw_tokens in self.category_to_raw_tokens.items():
            result = self._get_wheel_and_level1_for_category(category)
            if result:
                wheel_name, _ = result
                wheel_conflicts = self.per_wheel_conflicts.get(wheel_name, set())
                safe_tokens = raw_tokens - wheel_conflicts
            else:
                safe_tokens = raw_tokens
            
            self.category_to_safe_tokens[category] = safe_tokens
    
    # ================================================================
    # Phase 0 主流程
    # ================================================================
    
    def build(self) -> Dict[str, Set[int]]:
        """执行完整的 Phase 0 构建流程"""
        all_categories = (
            self.wheel_words_by_level["level1"] |
            self.wheel_words_by_level["level2"] |
            self.wheel_words_by_level["level3"]
        )
        
        print("="*70)
        print("Phase 0: 离线预处理 - 构建情感 Token 映射表")
        print("="*70)
        
        print(f"\n配置: mapping_depth='{self.mapping_depth}'")
        print(f"类别总数: {len(all_categories)}")
        
        print("\n[Step 1] 词库展开...")
        for category in all_categories:
            words = self._expand_category_to_words(category)
            self.category_to_words[category] = words
        
        print("[Step 2] 首 Token 提取与变体覆盖...")
        for category, words in self.category_to_words.items():
            all_tokens = set()
            for word in words:
                tokens = self._get_first_token_variants(word)
                all_tokens.update(tokens)
            self.category_to_raw_tokens[category] = all_tokens
        
        print("[Step 3] 互斥性清洗...")
        self._compute_global_token_counts()
        self._identify_conflicting_tokens()
        self._clean_tokens()
        
        print("[Step 4] 生成统计信息...")
        self._collect_stats()
        
        print("\n构建完成！")
        return self.category_to_safe_tokens
    
    def _collect_stats(self):
        self.stats.clear()
        
        for category in sorted(self.category_to_words.keys()):
            words = self.category_to_words.get(category, set())
            raw_tokens = self.category_to_raw_tokens.get(category, set())
            safe_tokens = self.category_to_safe_tokens.get(category, set())
            removed = raw_tokens - safe_tokens
            
            removed_examples = []
            for tid in list(removed)[:5]:
                decoded = self.tokenizer.decode([tid])
                removed_examples.append(f"{tid}:'{decoded}'")
            
            stat = TokenMapStats(
                category=category,
                words_before_filter=len(words),
                tokens_before_filter=len(raw_tokens),
                tokens_after_filter=len(safe_tokens),
                removed_tokens=len(removed),
                removed_token_examples=removed_examples,
            )
            self.stats.append(stat)
    
    def print_stats(self, show_all: bool = False):
        """打印统计信息"""
        print("\n" + "="*70)
        print("Token 映射统计")
        print("="*70)
        
        # 每个 wheel 的统计
        print("\n【每个情绪轮的冲突统计】")
        for wheel_name in sorted(self.per_wheel_conflicts.keys()):
            conflicts = self.per_wheel_conflicts[wheel_name]
            # 从 wheel_structure 获取该 wheel 的 Level1 类别
            wheel_l1s = list(self.wheel_structure.get(wheel_name, {}).keys())
            print(f"  {wheel_name}: Level1={sorted(wheel_l1s)}, 冲突Token={len(conflicts)}")
        
        total_words = 0
        total_before = 0
        total_after = 0
        
        level1_cats = sorted(self.wheel_words_by_level["level1"])
        
        print(f"\n{'类别':<20} {'词数':>8} {'过滤前':>10} {'过滤后':>10} {'移除':>8} {'保留率':>8}")
        print("-"*70)
        
        print("\n[Level1 核心情绪]")
        for stat in self.stats:
            if stat.category in level1_cats:
                total_words += stat.words_before_filter
                total_before += stat.tokens_before_filter
                total_after += stat.tokens_after_filter
                
                rate = stat.tokens_after_filter / stat.tokens_before_filter * 100 if stat.tokens_before_filter > 0 else 0
                print(f"  {stat.category:<18} {stat.words_before_filter:>8} {stat.tokens_before_filter:>10} {stat.tokens_after_filter:>10} {stat.removed_tokens:>8} {rate:>7.1f}%")
        
        print("-"*70)
        rate = total_after / total_before * 100 if total_before > 0 else 0
        print(f"  {'Level1 总计':<18} {total_words:>8} {total_before:>10} {total_after:>10} {total_before-total_after:>8} {rate:>7.1f}%")
        
        # 显示冲突 Token 示例（按 wheel 分组）
        print("\n【冲突 Token 示例 (每个wheel前3个)】")
        for wheel_name in sorted(self.per_wheel_conflicts.keys()):
            conflicts = self.per_wheel_conflicts[wheel_name]
            print(f"  {wheel_name} ({len(conflicts)} 个冲突):")
            for tid in list(conflicts)[:3]:
                decoded = self.tokenizer.decode([tid])
                print(f"    {tid}: '{decoded}'")
    
    def _build_wheel_level1_safe_tokens(self) -> Dict[str, Dict[str, Set[int]]]:
        """
        构建每个 wheel 内每个 Level1 类别的安全 Token 集合
        
        Returns:
            {wheel_name: {level1_category: set(safe_token_ids)}}
        """
        wheel_l1_tokens: Dict[str, Dict[str, Set[int]]] = {}
        
        for wheel_name, l1_data in self.wheel_structure.items():
            wheel_conflicts = self.per_wheel_conflicts.get(wheel_name, set())
            wheel_l1_tokens[wheel_name] = {}
            
            for l1 in l1_data.keys():
                # 收集该 L1 下所有词
                l1_words = set([l1])
                for l2, l3_words in l1_data[l1].items():
                    l1_words.add(l2)
                    l1_words.update(l3_words)
                
                # 获取所有 tokens 并移除冲突
                l1_tokens = set()
                for word in l1_words:
                    if word in self.category_to_raw_tokens:
                        l1_tokens.update(self.category_to_raw_tokens[word])
                
                safe_tokens = l1_tokens - wheel_conflicts
                wheel_l1_tokens[wheel_name][l1] = safe_tokens
        
        return wheel_l1_tokens
    
    def save(self, path: str):
        """保存映射表到文件"""
        # 构建每个 wheel 的 Level1 安全 Token 映射
        wheel_l1_safe_tokens = self._build_wheel_level1_safe_tokens()
        
        data = {
            "mapping_depth": self.mapping_depth,
            "wheel_words_by_level": {
                k: list(v) for k, v in self.wheel_words_by_level.items()
            },
            "categories": {
                cat: list(tokens) for cat, tokens in self.category_to_safe_tokens.items()
            },
            "category_to_words": {
                cat: list(words) for cat, words in self.category_to_words.items()
            },
            "wheel_structure": {
                wheel: {
                    l1: {l2: l3_words for l2, l3_words in l2_data.items()}
                    for l1, l2_data in l1_data.items()
                }
                for wheel, l1_data in self.wheel_structure.items()
            },
            "wheel_l1_safe_tokens": {
                wheel: {l1: list(tokens) for l1, tokens in l1_tokens.items()}
                for wheel, l1_tokens in wheel_l1_safe_tokens.items()
            },
            "wheel_reverse_index": {
                cat: list(info) for cat, info in self.wheel_reverse_index.items()
            },
            "per_wheel_conflicts": {
                wheel: list(conflicts) for wheel, conflicts in self.per_wheel_conflicts.items()
            },
            "conflicting_tokens": list(self.conflicting_tokens),
            "stats": [
                {
                    "category": s.category,
                    "words_before_filter": s.words_before_filter,
                    "tokens_before_filter": s.tokens_before_filter,
                    "tokens_after_filter": s.tokens_after_filter,
                    "removed_tokens": s.removed_tokens,
                }
                for s in self.stats
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n映射表已保存到: {path}")
    
    @classmethod
    def load(cls, path: str, tokenizer) -> "EmotionTokenMapBuilder":
        """从文件加载映射表"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        builder = cls(tokenizer=tokenizer, mapping_depth=data["mapping_depth"])
        builder.category_to_safe_tokens = {
            cat: set(tokens) for cat, tokens in data["categories"].items()
        }
        builder.conflicting_tokens = set(data["conflicting_tokens"])
        builder.wheel_words_by_level = {
            k: set(v) for k, v in data.get("wheel_words_by_level", {}).items()
        }
        
        # 加载 category_to_words
        if "category_to_words" in data:
            builder.category_to_words = {
                cat: set(words) for cat, words in data["category_to_words"].items()
            }
        
        # 加载 wheel 相关信息
        if "wheel_structure" in data:
            builder.wheel_structure = data["wheel_structure"]
        if "wheel_l1_safe_tokens" in data:
            builder.wheel_l1_safe_tokens = {
                wheel: {l1: set(tokens) for l1, tokens in l1_tokens.items()}
                for wheel, l1_tokens in data["wheel_l1_safe_tokens"].items()
            }
        if "wheel_reverse_index" in data:
            builder.wheel_reverse_index = {
                cat: tuple(info) for cat, info in data["wheel_reverse_index"].items()
            }
        if "per_wheel_conflicts" in data:
            builder.per_wheel_conflicts = {
                wheel: set(conflicts) for wheel, conflicts in data["per_wheel_conflicts"].items()
            }
        
        return builder


class EmotionTokenMap:
    """
    运行时情感 Token 映射表
    
    提供 Phase 1 在线训练所需的接口
    
    核心设计：
    - 每个情绪轮独立计算，轮内 Level1 类别之间不冲突
    - GT 映射到 n 个情绪轮时，分别计算后平均
    """
    
    def __init__(
        self,
        tokenizer,
        category_to_safe_tokens: Dict[str, Set[int]],
        level1_categories: Optional[Set[str]] = None,
        wheel_l1_safe_tokens: Optional[Dict[str, Dict[str, Set[int]]]] = None,
        wheel_structure: Optional[Dict] = None,
        wheel_reverse_index: Optional[Dict[str, Tuple]] = None,
        category_to_words: Optional[Dict[str, Set[str]]] = None,
    ):
        self.tokenizer = tokenizer
        self.category_to_safe_tokens = category_to_safe_tokens
        self.level1_categories = level1_categories or set()
        
        # Wheel 级别的映射（核心）
        self.wheel_l1_safe_tokens = wheel_l1_safe_tokens or {}  # {wheel: {l1: set(tokens)}}
        self.wheel_structure = wheel_structure or {}  # {wheel: {l1: {l2: [l3]}}}
        self.wheel_reverse_index = wheel_reverse_index or {}  # {word: (wheel, l1, l2, l3)}
        self.category_to_words = category_to_words or {}  # {category: set(words)} 包含同义词扩展
        
        # 构建词到类别的反向索引（用于 find_gt_in_wheels）
        self.word_to_categories: Dict[str, Set[str]] = defaultdict(set)
        for cat, words in self.category_to_words.items():
            for word in words:
                self.word_to_categories[word.lower()].add(cat)
        
        # 反向索引
        self.token_to_categories: Dict[int, Set[str]] = defaultdict(set)
        for cat, tokens in category_to_safe_tokens.items():
            for tid in tokens:
                self.token_to_categories[tid].add(cat)
        
        self.all_safe_tokens: Set[int] = set()
        for tokens in category_to_safe_tokens.values():
            self.all_safe_tokens.update(tokens)
        
        self.comma_token = self.tokenizer.encode(",", add_special_tokens=False)[0]
    
    def is_valid_emotion_token(self, token_id: int) -> bool:
        return token_id in self.all_safe_tokens
    
    def get_safe_tokens_for_category(self, category: str) -> Set[int]:
        return self.category_to_safe_tokens.get(category.lower().strip(), set())
    
    def get_categories_for_token(self, token_id: int) -> Set[str]:
        return self.token_to_categories.get(token_id, set())
    
    def get_all_categories(self) -> List[str]:
        return list(self.category_to_safe_tokens.keys())
    
    def get_all_wheels(self) -> List[str]:
        """获取所有情绪轮名称"""
        return list(self.wheel_l1_safe_tokens.keys())
    
    def get_wheel_level1_categories(self, wheel_name: str) -> List[str]:
        """获取指定情绪轮的所有 Level1 类别"""
        if wheel_name in self.wheel_l1_safe_tokens:
            return list(self.wheel_l1_safe_tokens[wheel_name].keys())
        return []
    
    def get_wheel_l1_safe_tokens(self, wheel_name: str, l1_category: str) -> Set[int]:
        """获取指定情绪轮的指定 Level1 类别的安全 Token"""
        if wheel_name in self.wheel_l1_safe_tokens:
            return self.wheel_l1_safe_tokens[wheel_name].get(l1_category.lower().strip(), set())
        return set()
    
    def find_gt_in_wheels(self, gt_words: List[str]) -> Dict[str, Set[str]]:
        """
        查找 GT 词汇在哪些情绪轮中，并返回对应的 Level1 类别
        
        策略：
        1. 通过 word_to_categories 找到词属于哪些类别（包含同义词扩展）
        2. 通过 wheel_reverse_index 找到这些类别对应的 wheel 和 Level1
        3. 同时也在原始 wheel_structure 中查找（作为备用）
        
        Args:
            gt_words: GT 情感词列表
        
        Returns:
            {wheel_name: set(level1_categories)}
        """
        wheel_to_l1: Dict[str, Set[str]] = defaultdict(set)
        
        for word in gt_words:
            word_lower = word.lower().strip()
            
            # 方法 1: 通过 word_to_categories 查找（包含同义词扩展）
            if word_lower in self.word_to_categories:
                categories = self.word_to_categories[word_lower]
                for category in categories:
                    # 通过 wheel_reverse_index 找到 wheel 和 Level1
                    if category in self.wheel_reverse_index:
                        wheel_info = self.wheel_reverse_index[category]
                        wheel_name, l1, l2, l3 = wheel_info
                        if l1:
                            wheel_to_l1[wheel_name].add(l1)
            
            # 方法 2: 在原始 wheel_structure 中查找（备用）
            for wheel_name, l1_data in self.wheel_structure.items():
                for l1, l2_data in l1_data.items():
                    # 检查是否是 Level1
                    if word_lower == l1.lower():
                        wheel_to_l1[wheel_name].add(l1)
                        continue
                    
                    # 检查是否是 Level2
                    if word_lower in [l2.lower() for l2 in l2_data.keys()]:
                        wheel_to_l1[wheel_name].add(l1)
                        continue
                    
                    # 检查是否是 Level3
                    for l2, l3_words in l2_data.items():
                        if word_lower in [w.lower() for w in l3_words]:
                            wheel_to_l1[wheel_name].add(l1)
                            break
        
        return dict(wheel_to_l1)
    
    def build_anchor_mask(
        self,
        answer_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase 1 Step 2: 动态锚点识别
        
        构建 Reward Calculation Mask M
        """
        if answer_token_ids.dim() > 1:
            answer_token_ids = answer_token_ids.squeeze()
        
        seq_len = answer_token_ids.size(0)
        mask = torch.zeros(seq_len, dtype=torch.long, device=answer_token_ids.device)
        
        for t in range(seq_len):
            token_id = answer_token_ids[t].item()
            
            # Check 1: 是否是词的开头
            is_word_start = (t == 0) or (answer_token_ids[t-1].item() == self.comma_token)
            
            # Check 2: 是否在 Clean_Vocab_Map 中
            # is_valid_token = self.is_valid_emotion_token(token_id)
            # is_valid_token = True
            
            # if is_word_start and is_valid_token:
            #     mask[t] = 1
            is_punctuation = (token_id == self.comma_token)  # 排除逗号本身

            if is_word_start and not is_punctuation:
                mask[t] = 1
        return mask
    
    def compute_category_log_prob(
        self,
        logits: torch.Tensor,
        category: str,
        epsilon: float = 1e-10,
    ) -> torch.Tensor:
        """
        计算类别的聚合置信度 S(c, t)
        
        S(c, t) = log(Σ_{id ∈ SafeTokens(c)} P(token_id | h_t) + ε)
        """
        is_1d = logits.dim() == 1
        if is_1d:
            logits = logits.unsqueeze(0)
        
        device = logits.device
        safe_tokens = self.get_safe_tokens_for_category(category)
        
        if not safe_tokens:
            result = torch.full((logits.size(0),), -100.0, device=device)
            return result.squeeze(0) if is_1d else result
        
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.tensor(list(safe_tokens), dtype=torch.long, device=device)
        category_probs = probs[:, token_ids]
        sum_probs = category_probs.sum(dim=-1)
        score = torch.log(sum_probs + epsilon)
        
        return score.squeeze(0) if is_1d else score


def create_emotion_token_map(
    tokenizer,
    emotion_wheel_root: Optional[str] = None,
    mapping_depth: str = "full",
    cache_path: Optional[str] = None,
) -> EmotionTokenMap:
    """
    创建情感 Token 映射表
    """
    if cache_path and os.path.exists(cache_path):
        print(f"从缓存加载映射表: {cache_path}")
        builder = EmotionTokenMapBuilder.load(cache_path, tokenizer)
    else:
        builder = EmotionTokenMapBuilder(
            tokenizer=tokenizer,
            emotion_wheel_root=emotion_wheel_root,
            mapping_depth=mapping_depth,
        )
        builder.build()
        builder.print_stats()
        
        if cache_path:
            builder.save(cache_path)
    
    # 构建 wheel 级别的 Token 映射
    wheel_l1_safe_tokens = {}
    if hasattr(builder, 'wheel_l1_safe_tokens'):
        wheel_l1_safe_tokens = builder.wheel_l1_safe_tokens
    elif hasattr(builder, '_build_wheel_level1_safe_tokens'):
        wheel_l1_safe_tokens = builder._build_wheel_level1_safe_tokens()
    
    return EmotionTokenMap(
        tokenizer=tokenizer,
        category_to_safe_tokens=builder.category_to_safe_tokens,
        level1_categories=builder.wheel_words_by_level.get("level1", set()),
        wheel_l1_safe_tokens=wheel_l1_safe_tokens,
        wheel_structure=getattr(builder, 'wheel_structure', {}),
        wheel_reverse_index=getattr(builder, 'wheel_reverse_index', {}),
        category_to_words=getattr(builder, 'category_to_words', {}),
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
        local_files_only=True,
        trust_remote_code=True
    )
    
    token_map = create_emotion_token_map(
        tokenizer=tokenizer,
        mapping_depth="full",
    )
    
    print("\n" + "="*70)
    print("测试 EmotionTokenMap")
    print("="*70)
    
    test_cats = ["happy", "angry", "sad", "scared"]
    for cat in test_cats:
        tokens = token_map.get_safe_tokens_for_category(cat)
        print(f"\n'{cat}': {len(tokens)} 个安全 Token")
        for tid in list(tokens)[:5]:
            print(f"  {tid} -> '{tokenizer.decode([tid])}'")
