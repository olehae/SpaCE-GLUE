from space_glue.data.base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset


class StepGame(BaseDataset):
    """
    StepGame Dataset
    """

    name: str = "StepGame"
    data_source: str = "data/stepgame_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        "You are a Spatial Reasoning Expert. Answer the question based on the story."
    )

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the StepGame dataset from a JSONL file.
        If that fails, it attempts to download it from HuggingFace using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from HuggingFace datasets and save locally
            ds = load_dataset("ZhengyanShi/StepGame", split="test")
            options = [opt.strip().lower() for opt in ds.unique("label")]

            # Add index to each item and save
            index = 0
            samples = {str(k): 0 for k in range(1, 11)}
            samples_per_k = 50  # k from 1 to 10 makes 500 samples
            with open(self.data_source, "w", encoding="utf-8") as f:
                for item in ds:
                    k = str(item["k_hop"])
                    if samples[k] >= samples_per_k:
                        continue
                    else:
                        samples[k] += 1
                        line = {
                            "index": index,
                            "k_hop": k,
                            "story": item["story"],
                            "question": item["question"],
                            "options": options,
                            "label": item["label"],
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        index += 1
                    yield line

    def to_prompt(self, item: Dict[str, Any]) -> str:
        """
        Converts a dataset item into a prompt string for the language model.

        Args:
            item: A dictionary representing a single data item.
        Returns:
            A string prompt for the language model.
        """
        return f"Story:\n{'\n'.join(item['story'])}\n\nQuestion:\n{item['question']}\n\nAnswer:"

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A dictionary mapping item indices to evaluation results (True/False).
        """
        ground_truth = item["label"]

        return [
            response.strip().lower() == ground_truth for response in item["responses"]
        ]

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracy.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracy.
        """
        if not dataset:
            raise ValueError("Dataset is empty. Cannot aggregate results.")
        scores = {str(k): [] for k in range(1, 11)}
        total_scores = []
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            scores[item["k_hop"]].append(mean_score)
            total_scores.append(mean_score)

        l = len(total_scores)
        acc = sum(total_scores) / l
        std = (
            (sum((s - acc) ** 2 for s in total_scores) / (l - 1)) ** 0.5 if l > 1 else 0
        )
        se = std / (l**0.5)

        return {
            "total": {
                "accuracy": acc,
                "standard_deviation": std,
                "standard_error": se,
                "count": l,
            },
            "by_difficulty": {
                k: {
                    "accuracy": (sum(v) / len(v) if len(v) > 0 else 0.0),
                    "count": len(v),
                }
                for k, v in scores.items()
            },
        }
