from .base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset
from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution


class SPaRC(BaseDataset):
    """
    SPaRC Dataset
    """

    name: str = "SPaRC"
    data_source: str = "data/sparc_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = r"You are an expert at solving puzzles games."

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the SPaRC dataset from a JSONL file.
        If that fails, it attempts to download it from HuggingFace using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from HuggingFace datasets and save locally
            ds = load_dataset("lkaesberg/SPaRC", "all", split="test")
            index = 0

            # Add index to each item and save
            with open(self.data_source, "w", encoding="utf-8") as f:
                for item in ds:
                    item = dict(item)
                    item["index"] = index
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    index += 1
                    yield item

    def to_prompt(self, item: Dict[str, Any]) -> str:
        """
        Converts a dataset item into a prompt string for the language model.

        Args:
            item: A dictionary representing a single data item.
        Returns:
            A string prompt for the language model.
        """
        return generate_prompt(item)

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A dictionary mapping item indices to evaluation results (True/False).
        """
        # See https://github.com/lkaesberg/SPaRC for evaluation details
        puzzle = item.copy()
        puzzle.pop("responses")
        results = []
        for response in item["responses"]:
            extracted_path = extract_solution_path(response, puzzle)
            is_correct = validate_solution(extracted_path, puzzle)
            results.append(is_correct)
        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracy.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracy.
        """
        overall = 0.0
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            overall += mean_score
        return {"accuracy": overall / len(dataset)}
