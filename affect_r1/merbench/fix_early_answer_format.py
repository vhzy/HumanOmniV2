#!/usr/bin/env python3
"""
Fix early answer format in existing result files.

This script reads a JSONL file with early answer results (where answer contains </answer> at the end),
and rewrites it with proper format by prepending <think> </think><answer> to raw_response.
"""

import argparse
import json
import re
from pathlib import Path


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_think_answer(raw_text: str):
    """Parse <think> and <answer> segments from a completion string."""
    raw_text = raw_text or ""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE)
    answer_match = ANSWER_PATTERN.search(raw_text)
    
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else raw_text.strip()
    valid = think_match is not None and answer_match is not None
    
    return {"think": think, "answer": answer, "valid_format": valid}


def fix_early_answer_record(record):
    """Fix a single record by prepending the prefilled content."""
    raw_response = record.get("raw_response", "")
    
    # Check if this looks like an early answer result (ends with </answer> but no opening <answer>)
    if "</answer>" in raw_response and not raw_response.strip().startswith("<answer>"):
        # Prepend the prefilled content
        fixed_response = "<think> </think><answer>" + raw_response
        
        # Re-parse with the fixed response
        parsed = parse_think_answer(fixed_response)
        
        # Update record
        record["raw_response"] = fixed_response
        record["think"] = parsed["think"]
        record["answer"] = parsed["answer"]
        record["has_valid_format"] = parsed["valid_format"]
        
        return True  # Fixed
    
    return False  # No fix needed


def main():
    parser = argparse.ArgumentParser(description="Fix early answer format in result files")
    parser.add_argument("input_file", type=str, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, default=None, 
                       help="Output JSONL file path (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be fixed without writing")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Read all records
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from {input_path}")
    
    # Fix records
    fixed_count = 0
    for record in records:
        if fix_early_answer_record(record):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} records")
    
    if args.dry_run:
        print("Dry run mode - no files written")
        if fixed_count > 0:
            print("\nExample of first fixed record:")
            for i, record in enumerate(records):
                if record.get("has_valid_format"):
                    print(f"  Name: {record['name']}")
                    print(f"  Raw response: {record['raw_response'][:100]}...")
                    print(f"  Think: {record['think'][:50]}...")
                    print(f"  Answer: {record['answer'][:100]}...")
                    break
    else:
        # Write fixed records
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Wrote fixed records to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

