# prepare_qa_jsonl.py

import os
import json

# 配置固定参数
image_root = "/mnt/storage/home/zy0058@students.ad.unt.edu/rrg/data/test_images"
prompt_path = "/mnt/storage/home/zy0058@students.ad.unt.edu/rrg/prompts/image.txt"
output_jsonl = "/mnt/storage/home/zy0058@students.ad.unt.edu/rrg/my_test_qa.jsonl"

# 加载 prompt
with open(prompt_path, "r") as f:
    prompt_text = f.read().strip()

# 开始构建
samples = []
question_id = 0

for folder in sorted(os.listdir(image_root)):
    folder_path = os.path.join(image_root, folder)
    if not os.path.isdir(folder_path):
        continue  # 跳过非文件夹

    # 默认只取 0.png
    image_filename = "0.png"
    image_path = os.path.join(folder_path, image_filename)

    if os.path.exists(image_path):
        samples.append({
            "question_id": str(question_id),
            "image": f"{folder}/{image_filename}",
            "question": prompt_text,
            "text": prompt_text
        })
        question_id += 1
    else:
        print(f"⚠️ Warning: {folder} 里没有找到 0.png，跳过。")

# 保存为 JSONL
os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
with open(output_jsonl, "w") as f:
    for entry in samples:
        f.write(json.dumps(entry) + "\n")

print(f"\n✅ Successfully generated {len(samples)} samples to {output_jsonl}")
