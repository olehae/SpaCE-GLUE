from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable, final

REQUIRED_FIELDS = {"index"}


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
    # All subcategories should be listed here. If not applicable, can be left empty.
    question_types: Dict[str, List[str]] = {}
    # Optional system prompt for the dataset
    system_prompt: str = ""

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
            )

    @final
    def __init__(self, **kwargs):
        """
        Loads dataset into self.data, then calls setup() for additional initialization.
        """
        self.setup(**kwargs)
        self.data: List[Dict[str, Any]] = list(self.load_data(**kwargs))
        self._validate_dataset_items()

    # ----------------------------------------------------------------------
    # Initialization hook — subclasses can optionally override this
    # ----------------------------------------------------------------------

    def setup(self, **kwargs):
        """
        Optional setup method.
        This method is called after load_data() and validation, allowing
        subclasses to initialize custom attributes or perform other setup
        without overriding __init__.

        Args:
            **kwargs: All keyword arguments passed to __init__
        """
        pass

    # ----------------------------------------------------------------------
    # Abstract methods — subclasses must implement these
    # ----------------------------------------------------------------------

    @abstractmethod
    def load_data(self) -> Iterable[Dict[str, Any]]:
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
    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            item (Dict[str, Any]): a single item containing at least:
                - index: int - the item index
                - responses: List[Any] - list of model responses

        Returns:
            List[Any]: evaluation scores for each response to the item
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
                - responses: List[Any] - list of model responses
                - scores: List[Any] - the evaluation scores (from evaluate())

        Returns:
            Dict[str, Any]: aggregated results, e.g., overall accuracy
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
            "system_prompt": self.system_prompt,
            "num_samples": len(self),
        }
