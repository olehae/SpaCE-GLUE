from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class defining the standard interface for LLM backends.
    """

    @abstractmethod
    def generate_single(self, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Generate a single response given a user and system prompt.
        """
        pass

    @abstractmethod
    def generate_batch(self, user_prompts: list, system_prompt: str = "You are a helpful assistant.") -> list:
        """
        Generate multiple responses using the same system prompt but varying user prompts.
        """
        pass
