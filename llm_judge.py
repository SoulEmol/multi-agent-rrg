import json
import random

# 读取我的 synthesis agent 结果文件
with open("/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/outputs_full_3/synthesis/synthesized_reports.json", "r") as f:
    data = json.load(f)

# 随机抽样 50 条
random.seed(42)
samples = random.sample(data, 50)

# 保存记录哪个是我的模型哪个是 baseline
id_to_model_map = {}

# 创建输出 prompt 文本文件
with open("/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/llm_judge/claude_llm_judge_prompts.txt", "w") as out_file:
    for i, item in enumerate(samples):
        gold = item["original_report"].strip()
        ours = item["synthesized_report"].strip()
        baseline = item["image_caption"].strip()
        img_id = item["image_id"]

        # 随机决定哪个是 Report A / Report B
        if random.random() > 0.5:
            report_a, report_b = ours, baseline
            model_a, model_b = "Ours", "Baseline"
        else:
            report_a, report_b = baseline, ours
            model_a, model_b = "Baseline", "Ours"

        id_to_model_map[f"sample_{i+1}"] = {
            "image_id": img_id,
            "Report A": model_a,
            "Report B": model_b
        }

        # 写 Claude prompt
        prompt = f"""You are an expert thoracic radiologist with extensive experience in interpreting chest X-rays. Your task is to evaluate and score the quality of two AI-generated radiology reports, based on five clinical and stylistic dimensions. You will be given:

- The original ground-truth radiology report (as reference).
- Report A: output from one model.
- Report B: output from another model.

For each dimension, assign a score from 0 to 10 (higher is better), and provide a short explanation for the score.

Evaluation Dimensions:
1. Clinically Significant Findings — Are key findings present in the reference report accurately retained?
2. Factual Consistency — Is the report factually consistent with the reference?
3. Diagnostic Alignment — Does it include appropriate diagnoses inferred from the findings?
4. Style Concordance — Does the writing style match the clinical, concise tone of the original?
5. Clarity_and_Conciseness — Is the report clear, well-structured, and not verbose?

Return your results in JSON format as follows:

{{
  "Report A": {{
    "Significant_findings": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Consistency": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Diagnosis": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Style": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Clarity_and_Conciseness": {{ "Score": <score>, "Reason": "<explanation>" }}
  }},
  "Report B": {{
    "Significant_findings": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Consistency": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Diagnosis": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Style": {{ "Score": <score>, "Reason": "<explanation>" }},
    "Clarity_and_Conciseness": {{ "Score": <score>, "Reason": "<explanation>" }}
  }},
  "Overall_Preference": "Report A or Report B"
}}

Ground Truth Report:
{gold}

Report A:
{report_a}

Report B:
{report_b}

---
"""
        out_file.write(f"========== Sample {i+1} ==========\n")
        out_file.write(prompt)
        out_file.write("\n\n")

# 保存 mapping 记录（用来标注结果）
with open("/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/llm_judge/llm_eval_mapping.json", "w") as f:
    json.dump(id_to_model_map, f, indent=2)
