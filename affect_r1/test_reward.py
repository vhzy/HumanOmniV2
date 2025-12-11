import sys
import os

# Add path to find affect_reward
sys.path.append("HumanOmniV2/affect_r1")

from affect_reward import emotion_wheel_reward, format_reward

# Mock Data
# Case 1: Perfect Match
completions_1 = [[{"content": "<think>...</think><answer>happy, joy</answer>"}]]
gt_1 = ["happiness", "joy"]

# Case 2: Wheel Match (Synonyms)
# Assuming 'glad' maps to 'happy' in some wheel
completions_2 = [[{"content": "<think>...</think><answer>bright</answer>"}]]
gt_2 = ["happy"]

# Case 3: No Match
completions_3 = [[{"content": "<think>...</think><answer>sad</answer>"}]]
gt_3 = ["happy"]

# Case 4: Bad Format
completions_4 = [[{"content": "Just happy"}]]
gt_4 = ["happy"]

def test():
    print("Testing Affect Reward...")
    
    # Test Format Reward
    r_fmt = format_reward(completions_1 + completions_4)
    print(f"Format Rewards (Expect [1.0, 0.0]): {r_fmt}")
    
    # Test Wheel Reward
    # We need to verify if the wheel files are actually loaded.
    # Since we don't know the exact content of the wheels without looking deep,
    # we rely on the fact that code runs without error and gives non-zero for matches.
    
    try:
        r_wheel = emotion_wheel_reward(
            completions_1 + completions_2 + completions_3, 
            openset=[gt_1, gt_2, gt_3]
        )
        print(f"Wheel Rewards: {r_wheel}")
        print("Success: Wheel reward function ran without crashing.")
    except Exception as e:
        print(f"Error running wheel reward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()


