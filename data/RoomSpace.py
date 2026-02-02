from .base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset


class RoomSpace(BaseDataset):
    """
    RoomSpace Dataset
    """

    name: str = "RoomSpace"
    data_source: str = "data/roomspace_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        "Analyze the spatial relationships between specified objects in a room, treating each object as a point within a 12x12 grid"
    )

    # The questions that will be asked per room layout
    questions = ["layout", "tpp", "o2", "o2_d2", "o2_layout"]

    # The options for the relation questions
    relation_options = [
        "north",
        "south",
        "east",
        "west",
        "north-east",
        "north-west",
        "south-east",
        "south-west",
        "overlap",
    ]

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the RoomSpace dataset from a JSONL file.
        If that fails, it attempts to download it from Huggingface using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from HuggingFace datasets and save locally
            # Remove Images and use top 50 samples to get 500 total questions
            ds = (
                load_dataset("Fangjun/RoomSpace", split="test", features=None)
                .remove_columns(["image_top_down", "image_north_facing"])
                .select(range(50))
            )
            all_data = []
            global_index = 0
            for item in ds:
                for question in self.questions:
                    # Two types of questions: yes/no and find relation
                    for q_suffix in ["yn", "fr"]:
                        line = {
                            "index": global_index,
                            "question_type": f"{question}_{q_suffix}",
                            "question": item[f"question_td_{question}_{q_suffix}"],
                            "options": (
                                self.relation_options
                                if q_suffix == "fr"
                                else ["Yes", "No"]
                            ),
                            "answer": item[f"answer_td_{question}_{q_suffix}"],
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
        return item["question"]

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A list of scores (final energy collected at start position).
        """
        model_answers = item["responses"]
        ground_truth = item["answer"]
        results = [ans in ground_truth for ans in model_answers]
        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracies.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracies.
        """
        yn_overall, yn_count = 0.0, 0
        fr_overall, fr_count = 0.0, 0
        categories = {q: [0.0, 0] for q in self.questions}  # [sum, count]
        total_overall, total_count = 0.0, 0
        for item in dataset:
            if item["question_type"].endswith("yn"):
                mean_score = sum(item["scores"]) / len(item["scores"])
                yn_overall += mean_score
                yn_count += 1
                categories[item["question_type"][:-3]][0] += mean_score
                categories[item["question_type"][:-3]][1] += 1
            else:
                mean_score = sum(item["scores"]) / len(item["scores"])
                fr_overall += mean_score
                fr_count += 1
                categories[item["question_type"][:-3]][0] += mean_score
                categories[item["question_type"][:-3]][1] += 1
            total_overall += mean_score
            total_count += 1

        yn_mean_acc = yn_overall / yn_count if yn_count > 0 else 0
        fr_mean_acc = fr_overall / fr_count if fr_count > 0 else 0
        total_mean_acc = total_overall / total_count if total_count > 0 else 0

        return {
            "total_accuracy": total_mean_acc,
            "total_count": total_count,
            "yn_accuracy": yn_mean_acc,
            "yn_count": yn_count,
            "fr_accuracy": fr_mean_acc,
            "fr_count": fr_count,
            "by_category": {
                k: {"accuracy": (v[0] / v[1] if v[1] > 0 else 0), "count": v[1]}
                for k, v in categories.items()
            },
        }
