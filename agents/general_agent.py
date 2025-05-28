# general_agent.py

import json
from agents.base_agent import BaseAgent

class GeneralAgent(BaseAgent):
    def __init__(self, config, model, prompt_path, name="general_agent"):
        super().__init__(config, model, prompt_path, name)

        # 加载 Top-K 检索结果
        retrieval_path = config.get("retrieval_output", "/mnt/storage/home/zy0058@students.ad.unt.edu/rrg/outputs/topk/out.json")
        with open(retrieval_path, "r") as f:
            self.retrieval_data = [json.loads(line) for line in f]

        # 映射 image_id -> reference_reports
        self.image_to_reports = {
            entry["image_id"]: entry["reference_reports"]
            for entry in self.retrieval_data
        }

    def run(self, image_id):
        # 获取该图像的 top-k 检索报告
        retrieved_reports = self.image_to_reports.get(image_id, [])

        # 构建 prompt
        formatted_prompt = self.format_prompt(
            retrieved_reports="\n\n".join(retrieved_reports)
        )

        # 使用模型生成报告
        output = self.model.generate(formatted_prompt)

        # 记录输入输出
        self.log(
            input_data={"image_id": image_id, "retrieved_reports": retrieved_reports},
            output_data=output
        )

        return output
