import json
from collections import defaultdict

# 文件路径（改成你自己的路径)
EVAL_RESULT_PATH = "/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/llm_judge/llm_eval_results.json"
MAPPING_PATH = "/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/llm_judge/llm_eval_mapping.json"

# 加载数据
with open(EVAL_RESULT_PATH, "r") as f:
    results = json.load(f)

with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

# 初始化统计结构
dimensions = ["Significant_findings", "Consistency", "Diagnosis", "Style", "Clarity_and_Conciseness"]
scores = defaultdict(lambda: defaultdict(list))
preferences = defaultdict(int)

# 用于排查未匹配项
unmatched = []

# 主处理流程
for sample in results:
    sid = sample.get("sample_id")
    if not sid or sid not in mapping:
        unmatched.append(sid or "<missing sample_id>")
        continue

    rep_a_model = mapping[sid]["Report A"]
    rep_b_model = mapping[sid]["Report B"]

    try:
        for dim in dimensions:
            scores[rep_a_model][dim].append(sample["Report A"][dim]["Score"])
            scores[rep_b_model][dim].append(sample["Report B"][dim]["Score"])

        pref = sample.get("Overall_Preference")
        if pref == "Report A":
            preferences[rep_a_model] += 1
        elif pref == "Report B":
            preferences[rep_b_model] += 1
    except Exception as e:
        unmatched.append(sid)

# 打印结果
print("\n=== LLM Evaluation Summary ===")
for model in scores:
    print(f"\nModel: {model}")
    for dim in dimensions:
        values = scores[model][dim]
        if values:
            avg = round(sum(values) / len(values), 2)
            print(f"  {dim:30s}: {avg}")
    print(f"  # Preferred: {preferences[model]}")

# 打印未成功处理的样本
if unmatched:
    print("\n=== Unmatched or Skipped Samples ===")
    for sid in unmatched:
        print("  -", sid)
else:
    print("\n✅ All samples processed successfully.")
