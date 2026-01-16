# Models using OpenAI-compatible API with vLLM structured output capabilities

from openai import AsyncOpenAI
from typing import List
from models.base_model import BaseModel
import asyncio


class OpenAIModel(BaseModel):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        temperature: float = None,
        reasoning_effort: str = None,
    ):
        """
        Initialize an OpenAI-compatible model.
        Works with both OpenAI and local vLLM endpoints.

        Args:
            model_name (str): Model name
            base_url (str): Base URL for the API endpoint
            api_key (str): API key for authentication
            temperature (float, optional): Sampling temperature
            reasoning_effort (str, optional): Reasoning effort level (e.g., "low", "medium", "high")
        """
        try:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self._name = name
            self.temperature = temperature
            self.reasoning_effort = reasoning_effort
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    @property
    def name(self) -> str:
        return self._name

    async def generate_single(
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

            # Build extra_body dict conditionally
            extra_body = {}
            if answer_options:
                extra_body["structured_outputs"] = {"choice": answer_options}
            if self.reasoning_effort:
                extra_body["reasoning_effort"] = self.reasoning_effort

            # Build kwargs for API call
            kwargs = {
                "model": self._name,
                "messages": messages,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if extra_body:
                kwargs["extra_body"] = extra_body

            # Make API call
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
