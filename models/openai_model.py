# Models using OpenAI-compatible API with vLLM structured output capabilities

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

    def generate_single(
        self,
        user_prompt: str,
        system_prompt: str,
        one_shot: dict = None,
        answer_options: List[str] = None,
    ) -> str:
        """
        Generate a single response given a user and system prompt.
        Args:
            user_prompt (str): The user prompt
            system_prompt (str): The system prompt
            one_shot (dict, optional): Dict with 'user' and 'assistant' keys for 1-shot example
            answer_options (list, optional): List of answer options
        Returns:
            str: The generated response
        """
        try:
            # Build messages list
            messages = [{"role": "system", "content": system_prompt}]

            # Add 1-shot example if provided
            if one_shot:
                messages.append({"role": "user", "content": one_shot["user"]})
                messages.append({"role": "assistant", "content": one_shot["assistant"]})

            # Add the actual user prompt
            messages.append({"role": "user", "content": user_prompt})

            if answer_options:
                response = self.client.chat.completions.create(
                    model=self._name,
                    temperature=self.temperature,
                    messages=messages,
                    extra_body={"guided_choice": answer_options},
                )
            else:
                response = self.client.chat.completions.create(
                    model=self._name,
                    temperature=self.temperature,
                    messages=messages,
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def generate_batch(
        self,
        dataset: List[dict],
        system_prompt: str,
        batch_size: int = 1,
        runs: int = 1,
    ) -> List[str]:
        """
        Return a list of generated responses for `dataset`.
        """
        raise NotImplementedError("Batch generation not implemented for OpenAIModel.")
