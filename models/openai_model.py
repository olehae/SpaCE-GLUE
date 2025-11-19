# Models using OpenAI-compatible API (local or cloud)

from openai import OpenAI
from typing import List
from models.base_model import BaseModel


class OpenAIModel(BaseModel):
    def __init__(
        self, name: str, base_url: str, api_key: str, temperature: float = 0.7
    ):
        """
        Initialize an OpenAI-compatible model.
        Works with both OpenAI and local vLLM endpoints.

        Args:
            model_name (str): Model name
            base_url (str): Base URL for the API endpoint
            api_key (str): API key for authentication
            temperature (float): Sampling temperature (default 0.7)
        """
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self._name = name
            self.temperature = temperature
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    @property
    def name(self) -> str:
        return self._name

    def generate_single(self, user_prompt: str, system_prompt: str) -> str:
        """
        Generate a single response given a user and system prompt.
        Args:
            user_prompt (str): The user prompt
            system_prompt (str): The system prompt
        Returns:
            str: The generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self._name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def generate_batch(self, user_prompts: List[str], system_prompt: str) -> List[str]:
        """
        Generate multiple responses using the same system prompt but varying user prompts.
        Args:
            user_prompts (list): List of user prompts
            system_prompt (str): The system prompt
        Returns:
            list: List of generated responses
        """
        try:
            responses = []
            for prompt in user_prompts:
                response = self.client.chat.completions.create(
                    model=self._name,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                responses.append(response.choices[0].message.content.strip())
            return responses
        except Exception as e:
            raise RuntimeError(f"OpenAI batch generation failed: {e}")
