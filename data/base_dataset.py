from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterable


class BaseDataset(ABC):
    """
    Abstract base class for all benchmark datasets.

    Subclasses should implement:
      - load_data()
      - to_prompt()
      - evaluate()
    """

    # Use a sentinel for required class attributes so we can enforce
    # that subclasses must provide concrete values at class-definition time.
    _REQUIRED = object()

    # Required attributes (subclasses must override these)
    name: str = _REQUIRED
    data_source: str = _REQUIRED
    batch_possible: bool = _REQUIRED

    # Optional attributes
    categories: Optional[List[str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = ("name", "data_source", "batch_possible")
        missing = [attr for attr in required if getattr(cls, attr, cls._REQUIRED) is cls._REQUIRED]
        if missing:
            raise TypeError(f"Class {cls.__name__} must define class attributes: {', '.join(missing)}")

    def __init__(self):
        """
        Args:
            **kwargs: Optional dataset-specific arguments.
        """
        self.data = self.load_data()

    # ----------------------------------------------------------
    # Abstract methods
    # ----------------------------------------------------------

    @abstractmethod
    def load_data(self) -> Iterable[Dict[str, Any]]:
        """Load the dataset and optionally filter / truncate it. 
        Returns the dataset as an iterable of dicts."""
        pass

    @abstractmethod
    def to_prompt(self, item: Dict[str, Any]) -> str:
        """Convert a single data item into a model prompt."""
        pass

    @abstractmethod
    def evaluate(self, item: Dict[str, Any], responses: str) -> float:
        """
        Compute the evaluation result for a model's response.

        Args:
            item (Dict[str, Any]): The original data
            response (str): The model's generated response
        Returns:
            float: The evaluation score for the response
        """
        pass

    # ----------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------

    def __iter__(self):
        """Iterate over dataset items."""
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def info(self) -> Dict[str, Any]:
        """Return metadata about the dataset."""
        return {
            "name": self.name,
            "data_source": self.data_source,
            "batch_possible": self.batch_possible,
            "categories": self.categories,
            "num_samples": len(self),
        }
