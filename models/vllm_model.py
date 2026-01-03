# Models from Huggingface using vLLM for local inference

from vllm import LLM, SamplingParams
from typing import List
from models.base_model import BaseModel


class VLLMModel(BaseModel):
    def __init__(self, name: str, temperature: float = 0.7):
        """
        Initialize a vLLM-based model for local inference.

        Args:
            name (str): Hugging Face model name (e.g., "meta-llama/Llama-3-8B-Instruct")
            temperature (float): Sampling temperature (default 0.7)
        """
        try:
            self.model = LLM(model=name)
            self._name = name
            self.temperature = temperature
        except Exception as e:
            raise RuntimeError(f"Failed to load model {name} with vLLM: {e}")

    @property
    def name(self) -> str:
        return self._name

    def generate_single(self, user_prompt: str, system_prompt: str) -> str:
        """
        Generate a single response given a user and system prompt.

        Args:
            user_prompt (str): The user's prompt
            system_prompt (str): The system prompt
        Returns:
            str: The generated response
        """
        try:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            sampling_params = SamplingParams(temperature=self.temperature)
            outputs = self.model.generate(
                [full_prompt], sampling_params=sampling_params
            )
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"vLLM single generation failed: {e}")

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
        raise NotImplementedError("Batch generation not implemented for VLLMModel.")
