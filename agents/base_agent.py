# agents/base_agent.py

import os
import json
from datetime import datetime

class BaseAgent:
    def __init__(self, config, model, prompt_path, name="base_agent"):
        self.config = config
        self.model = model
        self.name = name

        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()

        self.log_dir = os.path.join("logs", self.name)
        os.makedirs(self.log_dir, exist_ok=True)

    def format_prompt(self, **kwargs):
        """
        Replace placeholders in the prompt template with actual content.
        """
        prompt = self.prompt_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt

    def log(self, input_data, output_data):
        """
        Log input and output for debugging and reproducibility.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data
        }

        log_path = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

    def run(self, **kwargs):
        """
        Format the prompt, generate output, and log the result.
        """
        prompt = self.format_prompt(**kwargs)
        output = self.model.generate(prompt)
        self.log(input_data=kwargs, output_data=output)
        return output

