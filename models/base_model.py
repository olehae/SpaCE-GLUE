from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Minimal abstract base for LLM model backends.

    Subclasses must provide a `name` (as a property or attribute) and
    implement `generate_single` and `generate_batch`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, unique identifier for the model."""
        raise NotImplementedError

    @abstractmethod
    def generate_single(
        self, user_prompt: str, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """
        Generate a single response given a `user_prompt` and `system_prompt`.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        user_prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
    ) -> List[str]:
        """
        Return a list of generated responses for `user_prompts`.
        """
        pass
