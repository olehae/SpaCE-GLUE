# Models from Huggingface using vLLM for local inference

from models.vllm_model import LLM, SamplingParams
from base_model import BaseModel

class VLLMModel(BaseModel):
    def __init__(self, model_name: str, temperature: float = 0.7):
        """
        Initialize a vLLM-based model for local inference.

        Args:
            model_name (str): Hugging Face model name (e.g., "meta-llama/Llama-3-8B-Instruct")
            temperature (float): Sampling temperature (default 0.7)
        """
        try:
            self.model_name = model_name
            self.temperature = temperature
            self.model = LLM(model=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name} with vLLM: {e}")

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
            outputs = self.model.generate([full_prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"vLLM single generation failed: {e}")

    def generate_batch(self, user_prompts: list, system_prompt: str) -> list:
        """
        Generate multiple responses using the same system prompt but varying user prompts.
        Args:
            user_prompts (list): List of user prompts
            system_prompt (str): The system prompt
        Returns:
            list: List of generated responses
        """
        try:
            sampling_params = SamplingParams(temperature=self.temperature)
            full_prompts = [f"{system_prompt}\n\nUser: {p}\nAssistant:" for p in user_prompts]
            outputs = self.model.generate(full_prompts, sampling_params=sampling_params)
            return [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            raise RuntimeError(f"vLLM batch generation failed: {e}")
