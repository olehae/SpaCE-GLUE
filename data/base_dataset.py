from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable, final

REQUIRED_FIELDS = {"index", "item"}


class BaseDataset(ABC):
    """
    Abstract base class for all benchmark datasets with support for subcategories.

    Subclasses should implement:
      - load_data()
      - to_prompt()
      - evaluate()
    """

    _REQUIRED = object()

    # Required (subclasses must define these)
    name: str = _REQUIRED
    data_source: str = _REQUIRED
    batch_possible: bool = _REQUIRED

    # Optional Metadata describing all possible question types per subcategory
    # All subcategories should be listed here.

    question_types: Dict[str, List[str]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = ("name", "data_source", "batch_possible")
        missing = [
            attr
            for attr in required
            if getattr(cls, attr, cls._REQUIRED) is cls._REQUIRED
        ]

        if missing:
            raise TypeError(
                f"Class {cls.__name__} must define class attributes: {', '.join(missing)}"
            )
        # Prevent subclasses from overriding __init__ so that BaseDataset
        # always performs data loading and validation.
        if "__init__" in cls.__dict__:
            raise TypeError(
                f"Class {cls.__name__} must not override `__init__`. "
                "Implement `load_data` and other abstract methods instead; "
                "BaseDataset.__init__ handles data loading and validation."
            )

    @final
    def __init__(self, **kwargs):
        """
        Loads dataset into self.data.
        """
        self.data: List[Dict[str, Any]] = list(self.load_data(**kwargs))
        self._validate_dataset_items()

    # ----------------------------------------------------------------------
    # Abstract methods — subclasses implement these
    # ----------------------------------------------------------------------

    @abstractmethod
    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Load the dataset and optionally filter / truncate it.
        Returns the dataset as an iterable of dicts.
        """
        pass

    @abstractmethod
    def to_prompt(self, item: Dict[str, Any]) -> str:
        """
        Convert a single data item into a model prompt.
        """
        pass

    @abstractmethod
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[int, Any]:
        """
        Evaluate using the full dataset of items + responses.

        This method receives the complete dataset with responses and can use
        context about subcategories to evaluate items differently based on their
        category.

        Args:
            dataset (List[Dict[str, Any]]): list of items, each containing at least:
                - index: int - the item index
                - item: Dict[str, Any] - the original dataset item
                - response: str - the model's response

        Returns:
            Dict[int, Any]: mapping from index to score for each item
        """
        pass

    @abstractmethod
    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results across subcategories.

        This method receives the full dataset including scores and computes
        category-wise and overall statistics.

        Args:
            dataset (List[Dict[str, Any]]): list of items, each containing at least:
                - index: int - the item index
                - item: Dict[str, Any] - the original dataset item
                - response: str - the model's response
                - score: Any - the evaluation score (from evaluate())

        Returns:
            Dict[str, Any]: aggregated results, typically including overall and
                           per-category statistics
        """
        pass

    # ----------------------------------------------------------------------
    # Validation logic
    # ----------------------------------------------------------------------

    def _validate_dataset_items(self):
        """
        Checks that each item contains at least the required fields.
        """
        for i, item in enumerate(self.data):
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(
                    f"Dataset '{self.name}' returned an item missing required fields "
                    f"{missing}. Offending item at index {i}: {item}"
                )

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    def __iter__(self):
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
            "question_types": self.question_types,
            "num_samples": len(self),
        }
