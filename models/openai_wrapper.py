# models/openai_wrapper.py

from openai import OpenAI
import os

class OpenAIModel:
    def __init__(self, api_key, model_name="gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.default_temperature = 0.1
        self.default_max_tokens = 512

    def generate(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful radiologist generating a clinical report based on similar cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[OpenAIModel] Error generating response: {e}")
            return "[ERROR]"
