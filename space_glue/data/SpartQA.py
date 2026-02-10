from space_glue.data.base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import requests
import zipfile
import io
import json
from datasets import load_dataset


class SpartQA(BaseDataset):
    """
    SpartQA Dataset
    """

    name: str = "SpartQA"
    data_source: str = "data/spartqa_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        "You are an expert at answering questions on spatial reasoning. Answer the question based on the provided story. If something is not described in the story, answer 'DK'."
    )

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the SpartQA dataset from a JSONL file.
        If that fails, it attempts to download it from the original source.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from zip file and save locally
            # SpartQA-Human Train Set has 613 samples
            base_url = "https://www.cse.msu.edu/~kordjams/data/SpartQA_Human.zip"
            file_name = "SpartQA_Human/human_train.json"
            response = requests.get(base_url)
            response.raise_for_status()

            # Open the zip file from bytes
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(file_name) as f:
                    json_content = json.load(f)["data"]

            all_data = []
            global_index = 0
            for item in json_content:
                story = item["story"][0]
                for question in item["questions"]:
                    options = question["candidate_answers"]
                    options = [opt.strip() for opt in options]
                    if question["q_type"] == "YN":
                        options = ["Yes", "No", "DK"]

                    # Map answer indices to actual answers and strip whitespace
                    answer = question["answer"]
                    if all(isinstance(a, int) for a in answer):
                        answer = [
                            options[a] if 0 <= a < len(options) else None
                            for a in answer
                        ]
                    answer = [a.strip() for a in answer]
                    # In some FB questions there is no correct answer so the dataset provides an empty answer list
                    if question["q_type"] == "FB":
                        options.append("None")
                        if not answer:
                            answer = ["None"]

                    line = {
                        "index": global_index,
                        "story": story,
                        "q_type": question["q_type"],
                        "reasoning_type": question["reasoning_type"],
                        "question": question["question"],
                        "answer": answer,
                        "options": options,
                    }
                    all_data.append(line)
                    global_index += 1

            with open(self.data_source, "w", encoding="utf-8") as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    yield item

    def to_prompt(self, item: Dict[str, Any]) -> str:
        """
        Converts a dataset item into a prompt string for the language model.

        Args:
            item: A dictionary representing a single data item.
        Returns:
            A string prompt for the language model.
        """
        return f"{item['story']}\n\n{item['question']}"

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A list of scores.
        """
        model_answers = item["responses"]
        ground_truth = item["answer"]
        # Some answers have multiple correct options
        results = [ans in ground_truth for ans in model_answers]
        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean scores.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracies.
        """
        if not dataset:
            raise ValueError("Dataset is empty. Cannot aggregate results.")
        l = len(dataset)
        scores = {"FB": [], "FR": [], "CO": [], "YN": []}
        total_scores = []
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            scores[item["q_type"]].append(mean_score)
            total_scores.append(mean_score)

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
            "by_category": {
                k: {"accuracy": (sum(v) / len(v) if v else 0), "count": len(v)}
                for k, v in scores.items()
            },
        }
