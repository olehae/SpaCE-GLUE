from .base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset


class GeoGramBench(BaseDataset):
    """
    GeoGramBench Dataset
    """

    name: str = "GeoGramBench"
    data_source: str = "data/geogrambench_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        r"Let's think step by step and output the final answer within \\boxed{}. Your answer should be a mathematical expression in LaTeX format."
    )

    def load_data(self) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the GeoGramBench dataset from a JSONL file.
        If that fails, it attempts to download it from HuggingFace using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from HuggingFace datasets and save locally
            ds = load_dataset("LiAuto-DSR/GeoGramBench", split="train")
            index = 0
            with open(self.data_source, "w", encoding="utf-8") as f:
                for item in ds:
                    line = {
                        "index": index,
                        "problem_type": item["problem_type"],
                        "category": item["category"],
                        "problem": item["problem"],
                        "geo_code": item["geo_code"],
                        "answer": item["answer"],
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
        return f"Problem statement:\n{item['problem']}\n\nGeometric Code:\n{item['geo_code']}"

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            item: A single dataset item with model responses.
        Returns:
            A List of evaluation results (True/False).
        """
        results = []
        model_answers = item["responses"]
        ground_truth = item["answer"].replace("$", "").strip()
        for answer in model_answers:
            # Answers are expected to be in \\boxed{} format
            parsed_answer = (
                answer.replace("$", "").split(r"\\boxed{")[-1].removesuffix("}").strip()
            )
            results.append(parsed_answer == ground_truth)

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
