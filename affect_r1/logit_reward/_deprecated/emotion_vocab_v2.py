"""
情绪词表管理模块 V2

功能：
1. 支持完整的三层情绪轮映射：
   - 第一层：format.csv - 词形变化映射 (abandoned -> abandon)
   - 第二层：synonym.xlsx - 近义词映射 (exposed -> uncovered)
   - 第三层：wheel1-5.xlsx - 情绪轮层级 (L3 -> L2 -> L1)

2. 支持三种映射深度：
   - wheel_only: 只使用情绪轮核心词 (253个)
   - wheel_synonym: 情绪轮 + 近义词扩展
   - full: 全部三层，包含所有词形变化 (推荐，覆盖最全)

3. Token 匹配策略：
   针对实际 answer 格式: <answer>word1, word2, word3</answer>
   - 第一个词无空格前缀，需要用无空格编码匹配
   - 后续词有空格前缀（逗号后），需要用带空格编码匹配
   - 因此同时索引两种编码形式

用法：
    from emotion_vocab_v2 import create_emotion_vocab_manager
    
    manager = create_emotion_vocab_manager(
        tokenizer=tokenizer,
        mapping_depth="full",  # 推荐，覆盖所有词形变化
    )
    
    # 获取 GT 对应的类别
    gt_categories = manager.get_gt_categories(["frustrated", "anxious"], level="level1")
    
    # 在 answer 序列中查找情感词
    found = manager.find_emotion_words_in_sequence(answer_token_ids)
    
    # 构建 emotion mask
    mask = manager.build_emotion_mask(answer_token_ids)
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

import pandas as pd
import torch


class EmotionVocabManagerV2:
    """
    情绪词表管理器 V2
    
    支持完整的三层情绪轮映射和精确的 token 匹配。
    """
    
    # 映射深度常量
    DEPTH_WHEEL_ONLY = "wheel_only"
    DEPTH_WHEEL_SYNONYM = "wheel_synonym"
    DEPTH_FULL = "full"
    
    def __init__(
        self,
        tokenizer,
        emotion_wheel_root: Optional[str] = None,
        mapping_depth: str = "full",
    ):
        """
        初始化情绪词表管理器
        
        Args:
            tokenizer: HuggingFace tokenizer
            emotion_wheel_root: 情绪轮数据根目录
            mapping_depth: 映射深度，可选:
                - "wheel_only": 只用情绪轮核心词
                - "wheel_synonym": 情绪轮 + 近义词 (推荐)
                - "full": 全部三层
        """
        self.tokenizer = tokenizer
        self.mapping_depth = mapping_depth
        
        # 默认路径
        if emotion_wheel_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            emotion_wheel_root = os.path.join(current_dir, "..", "emotion_wheel")
        self.emotion_wheel_root = emotion_wheel_root
        
        # ============ 第一层: format.csv ============
        # word -> canonical_name (多个词形变化指向同一规范名)
        self.format_to_canonical: Dict[str, str] = {}
        # canonical_name -> {all_format_variants}
        self.canonical_to_formats: Dict[str, Set[str]] = defaultdict(set)
        
        # ============ 第二层: synonym.xlsx ============
        # word -> {synonyms} (双向映射)
        self.word_to_synonyms: Dict[str, Set[str]] = defaultdict(set)
        
        # ============ 第三层: wheel1-5.xlsx ============
        # wheel 结构: wheel_name -> {level1 -> {level2 -> [level3s]}}
        self.wheel_structure: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        # 反向索引: word -> (wheel_name, level1, level2, level3)
        self.wheel_reverse_index: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}
        # 按层级分类
        self.wheel_words_by_level: Dict[str, Set[str]] = {
            "level1": set(), "level2": set(), "level3": set()
        }
        
        # ============ Token 索引 ============
        # token_tuple -> {categories} (支持多 token 词)
        self.token_seq_to_categories: Dict[Tuple[int, ...], Set[str]] = defaultdict(set)
        # first_token_id -> {categories} (用于快速判断)
        self.first_token_to_categories: Dict[int, Set[str]] = defaultdict(set)
        # category -> {words}
        self.category_to_words: Dict[str, Set[str]] = defaultdict(set)
        # category -> {token_tuples}
        self.category_to_token_seqs: Dict[str, Set[Tuple[int, ...]]] = defaultdict(set)
        # 所有情感相关的 token 序列
        self.all_emotion_token_seqs: Set[Tuple[int, ...]] = set()
        # 所有情感相关的首 token
        self.all_first_tokens: Set[int] = set()
        
        # 统计信息
        self.stats = {}
        
        # 加载数据并构建索引
        self._load_all_data()
        self._build_token_index()
    
    # ================================================================
    # 数据加载
    # ================================================================
    
    def _load_format_csv(self):
        """加载第一层: format.csv (词形变化映射)"""
        csv_path = os.path.join(self.emotion_wheel_root, "format.csv")
        if not os.path.exists(csv_path):
            print(f"[Warning] format.csv not found: {csv_path}")
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
        """加载第二层: synonym.xlsx (近义词映射)"""
        xlsx_path = os.path.join(self.emotion_wheel_root, "synonym.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"[Warning] synonym.xlsx not found: {xlsx_path}")
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
        """加载第三层: wheel1-5.xlsx (情绪轮层级)"""
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
        """加载所有数据"""
        self._load_format_csv()
        self._load_synonym_xlsx()
        self._load_wheel_xlsx()
        
        all_wheel_words = (
            self.wheel_words_by_level["level1"] |
            self.wheel_words_by_level["level2"] |
            self.wheel_words_by_level["level3"]
        )
        self.stats.update({
            "format_entries": len(self.format_to_canonical),
            "synonym_entries": len(self.word_to_synonyms),
            "wheel_words_total": len(all_wheel_words),
            "wheel_level1_count": len(self.wheel_words_by_level["level1"]),
            "wheel_level2_count": len(self.wheel_words_by_level["level2"]),
            "wheel_level3_count": len(self.wheel_words_by_level["level3"]),
        })
    
    # ================================================================
    # 三层映射逻辑
    # ================================================================
    
    def map_word_to_wheel(
        self, word: str
    ) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        将输入词通过三层映射到情绪轮
        
        映射路径: input_word -> format归一化 -> synonym扩展 -> wheel定位
        
        Args:
            word: 输入词 (如 "frustrated", "overwhelmed")
        
        Returns:
            (wheel_name, level1, level2, level3) 或 None
            例如: ("wheel2", "angry", "frustrated", "annoyed")
        """
        word = word.lower().strip()
        
        # Step 1: format 归一化
        if word in self.format_to_canonical:
            candidates = [self.format_to_canonical[word]]
        else:
            candidates = [word]
        
        # Step 2: synonym 扩展
        stage2 = set()
        for cand in candidates:
            stage2.add(cand)
            if cand in self.word_to_synonyms:
                stage2.update(self.word_to_synonyms[cand])
        
        # Step 3: wheel 查找 (按字母序取第一个匹配)
        for candidate in sorted(stage2):
            if candidate in self.wheel_reverse_index:
                return self.wheel_reverse_index[candidate]
        
        return None
    
    def get_wheel_category(self, word: str, level: str = "level1") -> Optional[str]:
        """
        获取词的情绪轮类别
        
        Args:
            word: 输入词
            level: "level1" (核心情绪), "level2" (中层), "level3" (外层)
        
        Returns:
            对应层级的类别词，或 None
        """
        result = self.map_word_to_wheel(word)
        if result is None:
            return None
        
        _, l1, l2, l3 = result
        
        if level == "level1":
            return l1
        elif level == "level2":
            return l2 if l2 else l1
        elif level == "level3":
            return l3 if l3 else (l2 if l2 else l1)
        return None
    
    def expand_category_to_words(self, category: str) -> Set[str]:
        """
        将情绪轮类别扩展为所有相关词汇
        
        扩展深度由 self.mapping_depth 控制:
        - wheel_only: 只返回情绪轮中的词
        - wheel_synonym: 加上近义词
        - full: 加上所有词形变化
        
        Args:
            category: 情绪轮中的类别词 (如 "happy", "angry")
        
        Returns:
            所有相关词汇的集合
        """
        category = category.lower().strip()
        
        if category not in self.wheel_reverse_index:
            return set()
        
        wheel_name, l1, l2, l3 = self.wheel_reverse_index[category]
        words = set()
        
        # 收集情绪轮中的相关词
        if wheel_name in self.wheel_structure:
            wheel = self.wheel_structure[wheel_name]
            
            if l1 == category and l2 is None:
                # category 是 level1，收集其下所有 level2 和 level3
                words.add(l1)
                if l1 in wheel:
                    for l2_word, l3_list in wheel[l1].items():
                        words.add(l2_word)
                        words.update(l3_list)
            elif l2 == category and l3 is None:
                # category 是 level2，收集其下所有 level3
                words.add(l2)
                if l1 in wheel and l2 in wheel[l1]:
                    words.update(wheel[l1][l2])
            elif l3 == category:
                # category 是 level3，只返回自己
                words.add(l3)
        
        if self.mapping_depth == self.DEPTH_WHEEL_ONLY:
            return words
        
        # 扩展近义词
        if self.mapping_depth in [self.DEPTH_WHEEL_SYNONYM, self.DEPTH_FULL]:
            expanded = set()
            for w in words:
                expanded.add(w)
                if w in self.word_to_synonyms:
                    expanded.update(self.word_to_synonyms[w])
            words = expanded
        
        # 扩展词形变化
        if self.mapping_depth == self.DEPTH_FULL:
            expanded = set()
            for w in words:
                expanded.add(w)
                if w in self.canonical_to_formats:
                    expanded.update(self.canonical_to_formats[w])
            words = expanded
        
        return words
    
    # ================================================================
    # Token 索引构建
    # ================================================================
    
    def _build_token_index(self):
        """
        构建 token 索引
        
        关键设计：同时索引两种编码形式
        - 无空格版本: "word" (用于第一个词，紧跟 > 后面)
        - 带空格版本: " word" (用于后续词，逗号空格后面)
        
        这是因为实际 answer 格式是:
        <answer>word1, word2, word3</answer>
        """
        all_categories = (
            self.wheel_words_by_level["level1"] |
            self.wheel_words_by_level["level2"] |
            self.wheel_words_by_level["level3"]
        )
        
        single_token_count = 0
        multi_token_count = 0
        indexed_words = set()
        
        for category in all_categories:
            words = self.expand_category_to_words(category)
            self.category_to_words[category] = words
            
            for word in words:
                # 索引两种编码形式
                # 1. 无空格版本 (第一个词)
                # 2. 带空格版本 (后续词)
                variants = [word, " " + word]
                
                for variant in variants:
                    encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                    if not encoded:
                        continue
                    
                    token_tuple = tuple(encoded)
                    
                    # 避免重复计数
                    if token_tuple not in indexed_words:
                        indexed_words.add(token_tuple)
                        if len(encoded) == 1:
                            single_token_count += 1
                        else:
                            multi_token_count += 1
                    
                    # 建立索引
                    self.token_seq_to_categories[token_tuple].add(category)
                    self.first_token_to_categories[encoded[0]].add(category)
                    self.category_to_token_seqs[category].add(token_tuple)
                    self.all_emotion_token_seqs.add(token_tuple)
                    self.all_first_tokens.add(encoded[0])
        
        self.stats.update({
            "total_categories": len(all_categories),
            "unique_token_sequences": len(self.all_emotion_token_seqs),
            "single_token_sequences": single_token_count,
            "multi_token_sequences": multi_token_count,
        })
    
    # ================================================================
    # 匹配接口
    # ================================================================
    
    def get_gt_categories(
        self,
        gt_words: List[str],
        level: str = "level1",
    ) -> List[str]:
        """
        将 GT 词列表映射为情绪轮类别
        
        Args:
            gt_words: GT 情感词列表 (如 ["frustrated", "anxious"])
            level: 映射到的层级 ("level1", "level2", "level3")
        
        Returns:
            映射后的类别列表 (去重)
        """
        categories = set()
        for word in gt_words:
            cat = self.get_wheel_category(word, level)
            if cat:
                categories.add(cat)
        return list(categories)
    
    def get_all_categories(self) -> List[str]:
        """获取所有情绪轮类别"""
        return list(
            self.wheel_words_by_level["level1"] |
            self.wheel_words_by_level["level2"] |
            self.wheel_words_by_level["level3"]
        )
    
    def get_token_ids_for_category(self, category: str) -> Set[int]:
        """
        获取类别对应的所有首 token ID
        
        用于 logit_scorer.py 中计算类别得分
        
        Args:
            category: 情绪类别名
        
        Returns:
            该类别所有词的首 token ID 集合
        """
        category = category.lower().strip()
        token_ids = set()
        
        token_seqs = self.category_to_token_seqs.get(category, set())
        for seq in token_seqs:
            if seq:
                # 只取首 token (用于 logit 打分)
                token_ids.add(seq[0])
        
        return token_ids
    
    def find_emotion_words_in_sequence(
        self,
        token_ids: torch.Tensor,
        max_word_len: int = 5,
    ) -> List[Tuple[int, int, Set[str]]]:
        """
        在 token 序列中查找情感词
        
        使用贪婪最长匹配策略
        
        Args:
            token_ids: (seq_len,) 的 token id 序列
            max_word_len: 最大词长度 (token 数)
        
        Returns:
            [(start_pos, end_pos, categories), ...]
        """
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        
        seq_len = token_ids.size(0)
        found = []
        
        i = 0
        while i < seq_len:
            matched = False
            # 贪婪匹配：从长到短尝试
            for length in range(min(max_word_len, seq_len - i), 0, -1):
                token_tuple = tuple(token_ids[i:i+length].tolist())
                categories = self.token_seq_to_categories.get(token_tuple, set())
                
                if categories:
                    found.append((i, i + length, categories))
                    i += length
                    matched = True
                    break
            
            if not matched:
                i += 1
        
        return found
    
    def build_emotion_mask(
        self,
        token_ids: torch.Tensor,
        skip_tokens: Optional[Set[int]] = None,
        max_word_len: int = 5,
    ) -> torch.Tensor:
        """
        为序列构建情感词 mask
        
        Args:
            token_ids: (batch_size, seq_len) 或 (seq_len,) 的 token id 序列
            skip_tokens: 需要跳过的 token id (如逗号等)
            max_word_len: 最大词长度
        
        Returns:
            mask: 与 token_ids 相同 shape，情感词位置为 1
        """
        if skip_tokens is None:
            skip_tokens = set()
            for skip_word in [",", " ", "\n"]:
                encoded = self.tokenizer.encode(skip_word, add_special_tokens=False)
                skip_tokens.update(encoded)
        
        is_1d = token_ids.dim() == 1
        if is_1d:
            token_ids = token_ids.unsqueeze(0)
        
        batch_size, seq_len = token_ids.shape
        mask = torch.zeros_like(token_ids, dtype=torch.long)
        
        for b in range(batch_size):
            found = self.find_emotion_words_in_sequence(token_ids[b], max_word_len)
            for start, end, _ in found:
                # 跳过分隔符
                skip = any(token_ids[b, t].item() in skip_tokens for t in range(start, end))
                if not skip:
                    mask[b, start:end] = 1
        
        if is_1d:
            mask = mask.squeeze(0)
        
        return mask
    
    def is_emotion_token(self, token_id: int) -> bool:
        """判断 token_id 是否可能是情感词的首 token"""
        return token_id in self.all_first_tokens
    
    def __repr__(self) -> str:
        return (
            f"EmotionVocabManagerV2(\n"
            f"  mapping_depth='{self.mapping_depth}',\n"
            f"  wheel_words={self.stats.get('wheel_words_total', 0)},\n"
            f"  categories={self.stats.get('total_categories', 0)},\n"
            f"  token_sequences={self.stats.get('unique_token_sequences', 0)},\n"
            f"  single_token={self.stats.get('single_token_sequences', 0)},\n"
            f"  multi_token={self.stats.get('multi_token_sequences', 0)}\n"
            f")"
        )


# ================================================================
# 工厂函数
# ================================================================

def create_emotion_vocab_manager(
    tokenizer,
    emotion_wheel_root: Optional[str] = None,
    mapping_depth: str = "full",
) -> EmotionVocabManagerV2:
    """
    创建情绪词表管理器
    
    Args:
        tokenizer: HuggingFace tokenizer
        emotion_wheel_root: 情绪轮数据路径
        mapping_depth: 映射深度
            - "wheel_only": 只用情绪轮核心词 (253个)
            - "wheel_synonym": 情绪轮 + 近义词
            - "full": 全部三层，包含词形变化 (推荐，默认)
    
    Returns:
        EmotionVocabManagerV2 实例
    """
    return EmotionVocabManagerV2(
        tokenizer=tokenizer,
        emotion_wheel_root=emotion_wheel_root,
        mapping_depth=mapping_depth,
    )
