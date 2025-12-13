#!/usr/bin/env python3
"""
检查 rl_with_extracted_clues.jsonl 中 openset 字段是否为空，
如果为空，用 extracted_clues.reasoning_emotions 填充。
"""
import json
from pathlib import Path

input_file = Path("/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/rl_with_extracted_clues.jsonl")
output_file = input_file.with_name("rl_with_extracted_clues_fixed.jsonl")

fixed_count = 0
total_count = 0
empty_both_count = 0  # openset 和 reasoning_emotions 都为空的情况

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        
        total_count += 1
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Warning] Line {line_num}: JSON 解析失败: {e}")
            fout.write(line + "\n")
            continue
        
        openset = data.get("openset")
        extracted_clues = data.get("extracted_clues", {})
        reasoning_emotions = extracted_clues.get("reasoning_emotions", [])
        
        # 检查 openset 是否为空（None、空列表、或不存在）
        openset_is_empty = (openset is None or openset == [] or openset == "")
        
        if openset_is_empty:
            if reasoning_emotions and len(reasoning_emotions) > 0:
                # 用 reasoning_emotions 填充 openset
                data["openset"] = reasoning_emotions
                fixed_count += 1
                print(f"[Fixed] Line {line_num}: openset 为空，已用 reasoning_emotions 填充: {reasoning_emotions}")
                print(f"        Path: {data.get('path', 'N/A')}")
            else:
                # 两者都为空，记录但不修改
                empty_both_count += 1
                print(f"[Warning] Line {line_num}: openset 和 reasoning_emotions 都为空!")
                print(f"          Path: {data.get('path', 'N/A')}")
        
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print("\n" + "=" * 60)
print(f"处理完成!")
print(f"  总样本数: {total_count}")
print(f"  修复数量 (openset 为空，已用 reasoning_emotions 填充): {fixed_count}")
print(f"  警告数量 (openset 和 reasoning_emotions 都为空): {empty_both_count}")
print(f"  输出文件: {output_file}")
print("=" * 60)

# 如果你确认修复结果正确，可以用下面的命令替换原文件：
# import shutil
# shutil.move(output_file, input_file)
