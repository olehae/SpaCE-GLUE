from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Minimal abstract base for LLM model backends.

    Subclasses must provide a `name` (as a property or attribute) and implement `generate_single`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, unique identifier for the model."""
        raise NotImplementedError

    @abstractmethod
    def generate_single(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        one_shot: dict = None,
        answer_options: List[str] = None,
    ) -> str:
        """
        Generate a single response given a `user_prompt`, `system_prompt`, optional `one_shot` example, and optional `answer_options`.
        """
        pass
