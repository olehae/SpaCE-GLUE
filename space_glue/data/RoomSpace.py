from space_glue.data.base_dataset import BaseDataset
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
        if not dataset:
            raise ValueError("Dataset is empty, cannot aggregate results.")
        l = len(dataset)
        acc = []
        fr = []
        yn = []
        categories = {
            q + suffix: [0.0, 0] for q in self.questions for suffix in ["_yn", "_fr"]
        }  # [sum, count]
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            acc.append(mean_score)
            if item["question_type"].endswith("yn"):
                yn.append(mean_score)
                categories[item["question_type"]][0] += mean_score
                categories[item["question_type"]][1] += 1
            else:
                fr.append(mean_score)
                categories[item["question_type"]][0] += mean_score
                categories[item["question_type"]][1] += 1

        yn_count = len(yn)
        fr_count = len(fr)
        total_mean_acc = sum(acc) / l
        yn_mean_acc = sum(yn) / yn_count if yn_count > 0 else 0
        fr_mean_acc = sum(fr) / fr_count if fr_count > 0 else 0
        total_std = (
            (sum((s - total_mean_acc) ** 2 for s in acc) / (l - 1)) ** 0.5
            if l > 1
            else 0
        )
        total_se = total_std / (l**0.5)
        yn_std = (
            (sum((s - yn_mean_acc) ** 2 for s in yn) / (yn_count - 1)) ** 0.5
            if yn_count > 1
            else 0
        )
        yn_se = yn_std / (yn_count**0.5) if yn_count > 0 else 0
        fr_std = (
            (sum((s - fr_mean_acc) ** 2 for s in fr) / (fr_count - 1)) ** 0.5
            if fr_count > 1
            else 0
        )
        fr_se = fr_std / (fr_count**0.5) if fr_count > 0 else 0

        return {
            "total": {
                "accuracy": total_mean_acc,
                "standard_deviation": total_std,
                "standard_error": total_se,
                "count": l,
            },
            "by_type": {
                "yn": {
                    "accuracy": yn_mean_acc,
                    "standard_deviation": yn_std,
                    "standard_error": yn_se,
                    "count": yn_count,
                },
                "fr": {
                    "accuracy": fr_mean_acc,
                    "standard_deviation": fr_std,
                    "standard_error": fr_se,
                    "count": fr_count,
                },
            },
            "by_category": {
                k: {"accuracy": (v[0] / v[1] if v[1] > 0 else 0), "count": v[1]}
                for k, v in categories.items()
            },
        }
