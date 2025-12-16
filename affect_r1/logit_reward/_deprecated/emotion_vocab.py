"""
情绪词表管理模块

负责:
1. 加载情绪轮同义词映射 (format.csv)
2. 构建 token_id -> category 的反向索引
3. 提供快速查询接口
"""

import os
from typing import Dict, List, Optional, Set
from functools import lru_cache

import pandas as pd
import torch


class EmotionVocabManager:
    """
    情绪词表管理器
    
    用于管理情绪轮中的情感词及其同义词，支持高效的 token_id 到类别的映射。
    """
    
    def __init__(
        self,
        tokenizer,
        emotion_wheel_root: Optional[str] = None,
        filter_single_token: bool = True,
    ):
        """
        初始化情绪词表管理器
        
        Args:
            tokenizer: HuggingFace tokenizer
            emotion_wheel_root: 情绪轮数据根目录，默认使用内置路径
            filter_single_token: 是否只保留单 token 的词（推荐 True）
        """
        self.tokenizer = tokenizer
        self.filter_single_token = filter_single_token
        
        # 默认路径
        if emotion_wheel_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            emotion_wheel_root = os.path.join(current_dir, "..", "emotion_wheel")
        self.emotion_wheel_root = emotion_wheel_root
        
        # 核心数据结构
        self.category_to_synonyms: Dict[str, List[str]] = {}  # category -> [words]
        self.category_to_token_ids: Dict[str, Set[int]] = {}  # category -> {token_ids}
        self.token_id_to_categories: Dict[int, Set[str]] = {}  # token_id -> {categories}
        self.all_emotion_token_ids: Set[int] = set()  # 所有情感词的 token_id
        
        # 加载并构建索引
        self._load_format_csv()
        self._build_token_index()
    
    def _load_format_csv(self):
        """加载 format.csv，构建 category -> synonyms 映射"""
        csv_path = os.path.join(self.emotion_wheel_root, "format.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"format.csv not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            category = str(row["name"]).lower().strip()
            synonyms_str = str(row["format"]) if pd.notna(row["format"]) else ""
            synonyms = [s.strip().lower() for s in synonyms_str.split(",") if s.strip()]
            
            # 确保 category 本身也在同义词列表中
            if category not in synonyms:
                synonyms.insert(0, category)
            
            self.category_to_synonyms[category] = synonyms
    
    def _build_token_index(self):
        """构建 token_id 索引"""
        for category, synonyms in self.category_to_synonyms.items():
            token_ids = set()
            
            for word in synonyms:
                # 编码词，不添加特殊 token
                encoded = self.tokenizer.encode(word, add_special_tokens=False)
                
                if self.filter_single_token:
                    # 只取单 token 词的首个 token
                    if len(encoded) >= 1:
                        first_token_id = encoded[0]
                        token_ids.add(first_token_id)
                        
                        # 更新反向索引
                        if first_token_id not in self.token_id_to_categories:
                            self.token_id_to_categories[first_token_id] = set()
                        self.token_id_to_categories[first_token_id].add(category)
                else:
                    # 保留所有 token
                    for tid in encoded:
                        token_ids.add(tid)
                        if tid not in self.token_id_to_categories:
                            self.token_id_to_categories[tid] = set()
                        self.token_id_to_categories[tid].add(category)
            
            self.category_to_token_ids[category] = token_ids
            self.all_emotion_token_ids.update(token_ids)
    
    def is_emotion_token(self, token_id: int) -> bool:
        """判断 token_id 是否为情感词"""
        return token_id in self.all_emotion_token_ids
    
    def get_categories_for_token(self, token_id: int) -> Set[str]:
        """获取 token_id 对应的情感类别（可能有多个）"""
        return self.token_id_to_categories.get(token_id, set())
    
    def get_token_ids_for_category(self, category: str) -> Set[int]:
        """获取某个情感类别对应的所有 token_id"""
        category = category.lower().strip()
        return self.category_to_token_ids.get(category, set())
    
    def get_all_categories(self) -> List[str]:
        """获取所有情感类别"""
        return list(self.category_to_synonyms.keys())
    
    def build_answer_emotion_mask(
        self,
        answer_token_ids: torch.Tensor,
        skip_tokens: Optional[Set[int]] = None,
    ) -> torch.Tensor:
        """
        为 answer 序列构建情感词 mask
        
        Args:
            answer_token_ids: (batch_size, seq_len) 或 (seq_len,) 的 token id 序列
            skip_tokens: 需要跳过的 token id（如逗号、and 等）
        
        Returns:
            mask: 与 answer_token_ids 相同 shape 的 mask，情感词位置为 1，其余为 0
        """
        if skip_tokens is None:
            # 默认跳过逗号和 "and"
            skip_tokens = set()
            for skip_word in [",", "and", " and", ".", " ", "\n"]:
                encoded = self.tokenizer.encode(skip_word, add_special_tokens=False)
                skip_tokens.update(encoded)
        
        # 确保输入是 2D
        is_1d = answer_token_ids.dim() == 1
        if is_1d:
            answer_token_ids = answer_token_ids.unsqueeze(0)
        
        batch_size, seq_len = answer_token_ids.shape
        mask = torch.zeros_like(answer_token_ids, dtype=torch.long)
        
        for b in range(batch_size):
            for t in range(seq_len):
                tid = answer_token_ids[b, t].item()
                if tid in skip_tokens:
                    continue
                if self.is_emotion_token(tid):
                    mask[b, t] = 1
        
        if is_1d:
            mask = mask.squeeze(0)
        
        return mask
    
    def __repr__(self) -> str:
        return (
            f"EmotionVocabManager("
            f"categories={len(self.category_to_synonyms)}, "
            f"unique_tokens={len(self.all_emotion_token_ids)})"
        )
