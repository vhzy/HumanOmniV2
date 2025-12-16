"""
调试 _extract_answer_region 的问题
"""

from transformers import AutoTokenizer

# 用户提供的测试文本
test_text = '''system
You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your emotion reasoning process between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
user
Here is a video, with the audio from the video.

The subtitle of this video is: <Subtitle>I'm worried that he will turn green with envy as soon as he sees someone stronger.</Subtitle>.
As an expert in the field of emotions, please focus on the facial expressions, body movements, tone, subtitle content, etc., in the video to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the video.
 Please provide the words to describe emotions within the  <answer> </answer> tags.
assistant
<think>In the text, the subtitle content says, "I worry that he will turn green as soon as he sees Li Da." Based on the described facial expression of the woman in the video, which is a mixture of surprise and disbelief, along with her body language indicating of urgency and importance, we can infer that this sentence may be her concern or worry about someone or a situation. Additionally, based on the audio clue describing the trembling and high-pitched voice of the individual, as well as the implication of being overwhelmed or under a lot of pressure, we can speculate that this sentence may carry a tone of concern or anxiety. Therefore, this sentence may express the woman's worry that someone or something may make Li Da jealous or angry, aligning with her overall display of tension and stress.</think><answer>worry, concern, anxiety, tension, stress</answer>'''


def debug_tokenization():
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    # Tokenize
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"总 Token 数: {len(token_ids)}")
    
    # 找 answer 相关的 tokens
    print("\n=== 查找 'answer' 相关的 tokens ===")
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if 'answer' in decoded.lower() or '</' in decoded or '><' in decoded:
            print(f"位置 {i}: ID={tid}, 内容='{decoded}'")
    
    # 看 140-160 范围内的 tokens
    print("\n=== 位置 135-160 的 tokens（围绕问题位置 145）===")
    for i in range(135, min(160, len(token_ids))):
        decoded = tokenizer.decode([token_ids[i]])
        print(f"位置 {i}: ID={token_ids[i]}, 内容='{decoded}'")
    
    # 测试 <answer> 和 </answer> 的编码
    print("\n=== 单独编码测试 ===")
    print(f"'<answer>' -> {tokenizer.encode('<answer>', add_special_tokens=False)}")
    print(f"'</answer>' -> {tokenizer.encode('</answer>', add_special_tokens=False)}")
    print(f"'answer' -> {tokenizer.encode('answer', add_special_tokens=False)}")
    print(f"'>' -> {tokenizer.encode('>', add_special_tokens=False)}")
    print(f"'><answer>' -> {tokenizer.encode('><answer>', add_special_tokens=False)}")
    print(f"'stress</answer>' -> {tokenizer.encode('stress</answer>', add_special_tokens=False)}")
    
    # 找到实际的 answer 区域
    print("\n=== 分析 answer 区域 ===")
    answer_start_text = "<answer>"
    answer_end_text = "</answer>"
    
    # 在原文中找位置
    text_start = test_text.find(answer_start_text)
    text_end = test_text.find(answer_end_text)
    print(f"原文中 '<answer>' 位置: {text_start}")
    print(f"原文中 '</answer>' 位置: {text_end}")
    print(f"Answer 内容: '{test_text[text_start+len(answer_start_text):text_end]}'")
    
    # 分析实际的 token 边界问题
    print("\n=== 详细分析 ===")
    # 找到包含 'answer>' 的 token
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if 'answer>' in decoded:
            print(f"找到 'answer>' 在位置 {i}: '{decoded}'")
            print(f"  下一个 token (位置 {i+1}): '{tokenizer.decode([token_ids[i+1]])}'")
            # 继续看后面的 tokens
            print(f"  后续 tokens:")
            for j in range(i+1, min(i+10, len(token_ids))):
                print(f"    位置 {j}: '{tokenizer.decode([token_ids[j]])}'")
            break


def simulate_find_answer():
    """模拟修复后的 _extract_answer_region 行为"""
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/afs/hanzhiyuan/huggingface/Qwen2.5-Omni-7B",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    import torch
    token_ids = torch.tensor(tokenizer.encode(test_text, add_special_tokens=False))
    seq_len = token_ids.size(0)
    
    print("\n=== 新方法：找所有 answer 标签 ===")
    
    start_positions = []
    end_positions = []
    
    for i in range(seq_len):
        decoded = tokenizer.decode([token_ids[i].item()])
        decoded_lower = decoded.lower().strip()
        
        if decoded_lower == 'answer':
            prev_decoded = ""
            if i > 0:
                prev_decoded = tokenizer.decode([token_ids[i-1].item()])
            
            next_decoded = ""
            if i + 1 < seq_len:
                next_decoded = tokenizer.decode([token_ids[i+1].item()])
            
            print(f"位置 {i}: 'answer', 前='{prev_decoded}', 后='{next_decoded}'")
            
            is_open_tag = False
            is_close_tag = False
            
            if '</' in prev_decoded or prev_decoded.endswith('/'):
                is_close_tag = True
                print(f"  -> 识别为 </answer>")
            elif prev_decoded.endswith('<') or prev_decoded.strip() == '<':
                if not prev_decoded.endswith('</') and not prev_decoded.endswith('/'):
                    is_open_tag = True
                    print(f"  -> 识别为 <answer>")
            elif '><' in prev_decoded:
                is_open_tag = True
                print(f"  -> 识别为 <answer> (via ><)")
            
            if is_open_tag and next_decoded.startswith('>'):
                if len(next_decoded.strip()) > 1:
                    start_positions.append(i + 1)
                    print(f"  -> 添加 start 位置: {i + 1} (>与内容合并)")
                else:
                    start_positions.append(i + 2)
                    print(f"  -> 添加 start 位置: {i + 2}")
            elif is_close_tag:
                end_pos = i - 1
                end_positions.append(end_pos)
                print(f"  -> 添加 end 位置: {end_pos}")
    
    print(f"\n所有 start 位置: {start_positions}")
    print(f"所有 end 位置: {end_positions}")
    
    if start_positions and end_positions:
        last_start = start_positions[-1]
        last_end = end_positions[-1]
        
        if last_end > last_start:
            print(f"\n最终提取: start={last_start}, end={last_end}")
            answer_tokens = token_ids[last_start:last_end]
            answer_text = tokenizer.decode(answer_tokens.tolist())
            print(f"Answer 内容: '{answer_text}'")
        else:
            print(f"\n警告：end ({last_end}) <= start ({last_start})")


if __name__ == "__main__":
    debug_tokenization()
    simulate_find_answer()
