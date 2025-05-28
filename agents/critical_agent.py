from models.openai_wrapper import OpenAIModel

class CriticalAgent:
    def __init__(self, config, model: OpenAIModel, prompt_path: str, name: str):
        self.config = config
        self.model = model
        self.name = name
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()

    def run(self, image_id: str, preliminary_report: str, retrieved_reports: list):
        prompt = self.prompt_template.format(
            preliminary_report=preliminary_report,
            retrieved_reports="\n".join(retrieved_reports)
        )

        response = self.model.generate(prompt)

        parsed_response = response.strip()

        return parsed_response
