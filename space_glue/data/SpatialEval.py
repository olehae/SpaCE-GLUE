from space_glue.data.base_dataset import BaseDataset
from space_glue.data.spatialeval_evaluator import (
    extract_answer_from_text_mazenav,
    extract_answer_from_text_spatialgrid,
    extract_answer_from_text_spatialmap,
)
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset


class SpatialEval(BaseDataset):
    """
    SpatialEval Dataset
    """

    name: str = "SpatialEval"
    data_source: str = "data/spatialeval_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        "You are an expert at answering questions on spatial reasoning. Answer the question with one of the provided options."
    )

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the SpatialEval dataset from a JSONL file.
        If that fails, it attempts to download it from HuggingFace using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from HuggingFace datasets and save locally
            ds = load_dataset("MilaWang/SpatialEval", "tqa", split="test")
            tasks = ["spatialmap", "mazenav", "spatialgrid", "spatialreal"]

            # Get 125 items per task (to have total of 500) and redo indexing
            index = 0
            with open(self.data_source, "w", encoding="utf-8") as f:
                for task in tasks:
                    task_items = ds.filter(lambda x: x["id"].startswith(task)).select(
                        range(125)
                    )
                    # Add index to each item and save
                    for item in task_items:
                        # rename 'id' to 'original_id' to avoid confusion
                        item["original_id"] = item.pop("id")
                        item = {"index": index, "task": task, **item}
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
        return item["text"]

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A dictionary mapping item indices to evaluation results (True/False).
        """
        results = []
        ground_truth = item["oracle_answer"].lower()
        for response in item["responses"]:
            if item["task"] == "spatialmap":
                clean_response = extract_answer_from_text_spatialmap(response)
            elif item["task"] == "mazenav":
                clean_response = extract_answer_from_text_mazenav(response)
            elif item["task"] == "spatialgrid":
                clean_response = extract_answer_from_text_spatialgrid(response)
            else:  # spatialreal
                clean_response = response
            model_answer = str(clean_response).lower()
            results.append(ground_truth.lower() in model_answer.lower())
        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracies.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracies.
        """
        if not dataset:
            raise ValueError("Dataset is empty. Cannot aggregate results.")
        l = len(dataset)
        total_accuracy = []
        scores = {
            "spatialmap": [],
            "mazenav": [],
            "spatialgrid": [],
            "spatialreal": [],
        }
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            scores[item["task"]].append(mean_score)
            total_accuracy.append(mean_score)

        acc = sum(total_accuracy) / l
        std = (
            (sum((s - acc) ** 2 for s in total_accuracy) / (l - 1)) ** 0.5
            if l > 1
            else 0
        )
        se = std / (l**0.5)

        return {
            "total": {
                "accuracy": acc,
                "standard_deviation": std,
                "standard_error": se,
                "count": l,
            },
            "by_category": {
                k: {"accuracy": (sum(v) / len(v) if v else 0), "count": len(v)}
                for k, v in scores.items()
            },
        }
