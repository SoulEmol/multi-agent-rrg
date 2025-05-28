# run_train_pipeline.py

import os
import json
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "❌ OPENAI_API_KEY not found"

# === Path Configs ===
DATA_DIR = "/mnt/storage/home/zy0058@students.ad.unt.edu/rrg"
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
SAMPLES_DIR = os.path.join(DATA_DIR, "data", "test_images")
QA_JSONL_PATH = os.path.join(DATA_DIR, "my_test_qa.jsonl")  # my_test_qa.jsonl是提前生成好的
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🚦 Pipeline started.\n")

# ----------------------------------------------------------------------------------
# STEP 1: Top-K Agent - Retrieval
# ----------------------------------------------------------------------------------
from agents.topk_agent import TopKAgent

topk_config_path = os.path.join(DATA_DIR, "configs", "topk_config.json")
with open(topk_config_path) as f:
    topk_config = json.load(f)

print("🔍 [1/5] Running TopKAgent...")
topk_agent = TopKAgent(topk_config)
topk_agent.run()
print(f"✅ Top-k results saved to: {topk_config['output_path']}\n")

# ----------------------------------------------------------------------------------
# STEP 2: General Agent - Generate preliminary report
# ----------------------------------------------------------------------------------
from agents.general_agent import GeneralAgent
from models.openai_wrapper import OpenAIModel

general_agent = GeneralAgent(
    config={},
    model=OpenAIModel(api_key=api_key, model_name="gpt-4o"),
    prompt_path=os.path.join(PROMPT_DIR, "general.txt"),
    name="general_agent"
)

with open(topk_config["output_path"]) as f:
    retrieval_data = [json.loads(line) for line in f]

print("📝 [2/5] Running GeneralAgent...")
prelim_output = []
for entry in retrieval_data:
    image_id = entry["image_id"]
    retrieved_reports = entry["reference_reports"]
    original_report = entry.get("original_report", "")
    
    general_agent.image_to_reports = {image_id: retrieved_reports}
    prelim = general_agent.run(image_id=image_id)

    prelim_output.append({
        "image_id": image_id,
        "original_report": original_report,
        "retrieved_reports": retrieved_reports,
        "preliminary_report": prelim
    })

general_output_path = os.path.join(OUTPUT_DIR, "general", "preliminary_reports.json")
os.makedirs(os.path.dirname(general_output_path), exist_ok=True)
with open(general_output_path, "w") as f:
    json.dump(prelim_output, f, indent=2)

print(f"✅ Preliminary reports saved to: {general_output_path}\n")

# ----------------------------------------------------------------------------------
# STEP 3: Critical Agent - Extract key findings
# ----------------------------------------------------------------------------------
from agents.critical_agent import CriticalAgent

critical_agent = CriticalAgent(
    config={},
    model=OpenAIModel(api_key=api_key, model_name="gpt-4o"),
    prompt_path=os.path.join(PROMPT_DIR, "critical.txt"),
    name="critical_agent"
)

print("🔎 [3/5] Running CriticalAgent...")
crit_output = []
for entry in prelim_output:
    result = critical_agent.run(
        image_id=entry["image_id"],
        preliminary_report=entry["preliminary_report"],
        retrieved_reports=entry["retrieved_reports"]
    )
    entry["critical_text"] = result
    crit_output.append(entry)

crit_output_path = os.path.join(OUTPUT_DIR, "critical", "critical_findings.json")
os.makedirs(os.path.dirname(crit_output_path), exist_ok=True)
with open(crit_output_path, "w") as f:
    json.dump(crit_output, f, indent=2)

print(f"✅ Critical findings saved to: {crit_output_path}\n")

# ----------------------------------------------------------------------------------
# STEP 4: Image Agent (LLaVA-Med) - Generate visual captions
# ----------------------------------------------------------------------------------
from agents.image_agent_llava_med import generate_image_captions_llava_med

print("🖼️ [4/5] Running ImageAgent (LLaVA-Med)...")

MODEL_PATH = "/mnt/storage/home/zy0058@students.ad.unt.edu/llava-med-v1.5-mistral-7b"
QA_JSONL_PATH = os.path.join(DATA_DIR, "my_test_qa.jsonl")
TEMP_ANSWER_JSONL = os.path.join(OUTPUT_DIR, "visual", "temp_outputs.jsonl")

# Step 4.1: 生成图片描述，保存到临时outputs
generate_image_captions_llava_med(
    model_path=MODEL_PATH,
    image_folder=SAMPLES_DIR,
    question_file=QA_JSONL_PATH,
    answers_file=TEMP_ANSWER_JSONL,
    conv_mode="mistral_instruct",
    temperature=0.0
)

# Step 4.2: 更新critical findings，加上image_caption
with open(crit_output_path, "r") as f:
    critical_findings = json.load(f)

# 加载qa_jsonl和生成的输出
with open(QA_JSONL_PATH, "r") as f:
    qa_entries = [json.loads(line) for line in f]
image_to_question = {e["image"]: e["question_id"] for e in qa_entries}

with open(TEMP_ANSWER_JSONL, "r") as f:
    vqa_outputs = [json.loads(line) for line in f]
qid_to_caption = {v["question_id"]: v["text"] for v in vqa_outputs}

# 插入caption
for entry in critical_findings:
    image_name = entry["image_id"]
    question_id = image_to_question.get(image_name)
    caption = qid_to_caption.get(str(question_id), "No findings.")
    entry["image_caption"] = caption

visual_output_path = os.path.join(OUTPUT_DIR, "visual", "critical_findings_with_caption.json")
os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)
with open(visual_output_path, "w") as f:
    json.dump(critical_findings, f, indent=2)

print(f"✅ Updated critical findings with image captions saved to: {visual_output_path}\n")

# ----------------------------------------------------------------------------------
# STEP 5: Summarizing Agent - Synthesize full report
# ----------------------------------------------------------------------------------
from agents.summarizing_agent import SummarizingAgent

summarizer = SummarizingAgent(
    config={},
    model=OpenAIModel(api_key=api_key, model_name="gpt-4o"),
    prompt_path=os.path.join(PROMPT_DIR, "summarizing_prompt.txt"),
    name="summarizing_agent"
)

synthesis_output_path = os.path.join(OUTPUT_DIR, "synthesis", "synthesized_reports.json")
print("🧠 [5/5] Running SummarizingAgent...")
summarizer.run_on_file(
    input_path=visual_output_path,
    output_path=synthesis_output_path
)

print(f"✅ Synthesized reports saved to: {synthesis_output_path}\n")
print("🎉 Full multi-agent pipeline completed successfully!\n")
