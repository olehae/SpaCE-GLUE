from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class defining the standard interface for LLM backends.

    Subclasses must define the class attribute `name`.
    """

    # Sentinel used to detect missing name values
    _REQUIRED = object()

    # Subclasses may provide a class-level default name, but instances must
    # always have `self.name` set. We perform the check in `__init__` so
    # that instance-level names (passed at construction) are supported.
    name: str = _REQUIRED

    def __init__(self, name: str | None = None):
        # If a name was provided to the constructor, use it.
        if name is not None:
            self.name = name
            return

        # Otherwise, fall back to a class-level name if present.
        cls_name = getattr(self.__class__, "name", self._REQUIRED)
        if cls_name is not self._REQUIRED and cls_name is not None:
            self.name = cls_name
            return

        raise TypeError(
            f"Instance of {self.__class__.__name__} requires a 'name' (pass name=...) or set a class attribute 'name')"
        )

    @abstractmethod
    def generate_single(
        self, user_prompt: str, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """
        Generate a single response given a user and system prompt.
        """
        pass

    @abstractmethod
    def generate_batch(
        self, user_prompts: list, system_prompt: str = "You are a helpful assistant."
    ) -> list:
        """
        Generate multiple responses using the same system prompt but varying user prompts.
        """
        pass
