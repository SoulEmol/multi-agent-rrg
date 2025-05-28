from models.openai_wrapper import OpenAIModel

class SummarizingAgent:
    def __init__(self, config, model, prompt_path, name="summarizing_agent"):
        self.config = config
        self.model = model
        self.name = name

        with open(prompt_path, "r") as f:
            self.prompt = f.read()

    def run_on_file(self, input_path, output_path):
        import json
        import os

        with open(input_path, "r") as f:
            data = json.load(f)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        outputs = []

        for item in data:
            final_prompt = self.prompt.format(
                image_caption=item.get("image_caption", ""),
                preliminary_report=item.get("preliminary_report", ""),
                critical_text=item.get("critical_text", "")
            )

            # ✅ 注意这里用 generate
            response = self.model.generate(final_prompt)
            item["synthesized_report"] = response
            outputs.append(item)

        with open(output_path, "w") as f:
            json.dump(outputs, f, indent=2)
