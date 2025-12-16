#!/usr/bin/env python3
"""
找出训练过程中特定steps加载的数据索引
"""
import torch
import json

# 训练配置参数
num_samples = 26101  # 数据总量
batch_size = 8  # 实际batch size (effective_batch_size // num_generations)
num_generations = 8  # 每个样本生成次数
seed = 42  # data_seed

# 目标步骤范围
start_step = 2300
end_step = 2620

def get_sampler_indices(num_samples, batch_size, num_generations, seed):
    """
    重现RepeatRandomSampler的采样逻辑
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # 生成随机排列
    indexes = torch.randperm(num_samples, generator=generator).tolist()
    
    # 按batch_size分组
    chunks = [indexes[i:i + batch_size] for i in range(0, len(indexes), batch_size)]
    
    # 过滤掉不完整的chunk
    chunks = [chunk for chunk in chunks if len(chunk) == batch_size]
    
    # 展开：每个index重复num_generations次
    all_indices = []
    for chunk in chunks:
        for index in chunk:
            for _ in range(num_generations):
                all_indices.append(index)
    
    return all_indices, chunks

# 获取完整的采样顺序
all_indices, chunks = get_sampler_indices(num_samples, batch_size, num_generations, seed)

# 每个step处理的样本数 = batch_size * num_generations = 64
samples_per_step = batch_size * num_generations

print(f"=" * 80)
print(f"训练配置信息")
print(f"=" * 80)
print(f"数据总量: {num_samples}")
print(f"实际batch size (unique样本): {batch_size}")
print(f"每个样本生成次数: {num_generations}")
print(f"每个step处理的总样本数: {samples_per_step}")
print(f"总steps数: {len(chunks)}")
print(f"随机种子: {seed}")
print(f"\n目标步骤范围: {start_step} - {end_step}")
print(f"=" * 80)

# 计算目标步骤范围对应的索引
start_idx = start_step * samples_per_step
end_idx = (end_step + 1) * samples_per_step

# 确保不超出范围
end_idx = min(end_idx, len(all_indices))

# 提取目标范围的索引
target_indices = all_indices[start_idx:end_idx]

# 统计unique数据索引
unique_indices = sorted(set(target_indices))

print(f"\n第 {start_step} 到 {end_step} 步的数据统计:")
print(f"-" * 80)
print(f"总样本数（包含重复）: {len(target_indices)}")
print(f"Unique数据索引数量: {len(unique_indices)}")
print(f"覆盖的step数: {end_step - start_step + 1}")

# 保存详细信息到文件
output_data = {
    "config": {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_generations": num_generations,
        "seed": seed,
        "total_steps": len(chunks)
    },
    "query_range": {
        "start_step": start_step,
        "end_step": end_step,
        "total_samples_with_repeats": len(target_indices),
        "unique_samples": len(unique_indices)
    },
    "unique_data_indices": unique_indices
}

# 按step组织数据
steps_detail = {}
for step in range(start_step, end_step + 1):
    if step >= len(chunks):
        break
    step_chunk = chunks[step]
    steps_detail[step] = {
        "unique_indices": step_chunk,
        "count": len(step_chunk),
        "repeated_times": num_generations
    }

output_data["steps_detail"] = steps_detail

# 保存到JSON文件
output_file = "/mnt/afs/hanzhiyuan/code/HumanOmniV2/affect_r1/data_indices_step_2300_2620.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n详细信息已保存到: {output_file}")
print(f"=" * 80)

# 打印前5个和后5个步骤的详细信息作为示例
print(f"\n示例：前5个步骤的数据索引")
print(f"-" * 80)
for step in range(start_step, min(start_step + 5, end_step + 1)):
    if step >= len(chunks):
        break
    print(f"Step {step}: {chunks[step]}")

print(f"\n示例：后5个步骤的数据索引")
print(f"-" * 80)
for step in range(max(start_step, end_step - 4), end_step + 1):
    if step >= len(chunks):
        break
    print(f"Step {step}: {chunks[step]}")

print(f"\n" + "=" * 80)
print(f"所有unique数据索引（前20个）: {unique_indices[:20]}")
if len(unique_indices) > 20:
    print(f"... 共 {len(unique_indices)} 个索引，完整列表见JSON文件")
print(f"=" * 80)

