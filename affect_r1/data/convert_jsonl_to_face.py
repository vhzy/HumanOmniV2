import json
import os
from tqdm import tqdm

input_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/sft_data.jsonl"
output_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data/sft_data_face.jsonl"

print(f"Processing {input_file} -> {output_file}")

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    lines = fin.readlines()
    for line in tqdm(lines):
        if not line.strip():
            continue
            
        data = json.loads(line)
        
        # Update 'video' path
        if "video" in data:
            src_path = data["video"]
            dirname = os.path.dirname(src_path)
            basename = os.path.basename(src_path)
            parent_dir = os.path.dirname(dirname)
            
            # Construct new path: parent / video_face / filename
            # This aligns with the logic in preprocess_face.py which creates 'video_face' as a sibling
            new_path = os.path.join(parent_dir, "video_face", basename)
            data["video"] = new_path
            
        # Update 'path' field if it exists and looks like a video path
        if "path" in data:
            # Usually path == video for video datasets
            src_path = data["path"]
            if src_path.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                dirname = os.path.dirname(src_path)
                basename = os.path.basename(src_path)
                parent_dir = os.path.dirname(dirname)
                
                new_path = os.path.join(parent_dir, "video_face", basename)
                data["path"] = new_path

        fout.write(json.dumps(data) + "\n")

print("Done.")

