from .base_dataset import BaseDataset
from typing import Dict, List, Iterable, Any
import json
from datasets import load_dataset
import re


class bAbI(BaseDataset):
    """
    bAbI Dataset
    """

    name: str = "bAbI"
    data_source: str = "data/babi_data.jsonl"
    batch_possible: bool = True

    system_prompts: dict = {
        "qa17": "You are a Spatial Reasoning Expert. Answer the question with yes or no based on the story.",
        "qa18": "You are a Spatial Reasoning Expert. Answer the question with yes or no based on the story.",
        "qa19": "You are a Spatial Reasoning Expert. Answer the question based on the story. Answer with the steps needed to go from the source location to the target location, each step can either be n, s, e, or w representing north, south, east, or west movement. The steps should be separated by commas. Answer only with the steps, do not add any additional text. A valid answer example is: n,e",
    }

    # This is needed to distribute samples evenly across tasks
    def distribute(self, task_num, cat_num):
        base = task_num // cat_num
        remainder = task_num % cat_num
        return [base + 1 if i < remainder else base for i in range(cat_num)]

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the bAbI dataset from a JSONL file.
        If that fails, it attempts to download it from HuggingFace using the datasets library.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            index = 0
            per_task = self.distribute(500, 3)  # 500 samples across 3 tasks
            with open(self.data_source, "w", encoding="utf-8") as f:
                for task_name, num in zip(["qa17", "qa18", "qa19"], per_task):
                    ds = load_dataset(
                        "facebook/babi_qa",
                        trust_remote_code=True,
                        type="en",
                        task_no=task_name,
                        split="test",
                    )
                    # number of samples per question
                    per_question = self.distribute(num, len(ds))
                    for item, count in zip(ds, per_question):
                        item = dict(item["story"])
                        story = []
                        question_added = 0
                        prev = None
                        for i, text in enumerate(item["text"]):
                            if question_added >= count:
                                # Only add the required number of questions
                                break
                            if text == prev:
                                # There are duplicates in the dataset, skip them
                                continue
                            if item["type"][i]:
                                line = {
                                    "index": index,
                                    "task": task_name,
                                    "story": story,
                                    "question": text,
                                    "system_prompt": self.system_prompts[task_name],
                                    "answer": item["answer"][i],
                                }
                                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                                prev = text
                                index += 1
                                question_added += 1
                                yield line
                            else:
                                story.append(item["text"][i])

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
        results = []
        ground_truth = item["answer"].lower()
        responses = []
        for resp in item["responses"]:
            if "</think>" in resp:
                # Extract the part after </think>
                resp = resp.split("</think>")[-1].strip()
            responses.append(resp.lower())
        # Task 17 and 18 are yes/no questions
        if item["task"] in ["qa17", "qa18"]:
            # Try to filter a yes/no answer from the response
            pattern = r"\b(yes|no)\b"
            for response in responses:
                match = re.search(pattern, response, flags=re.I)
                if match:
                    predicted = match.group().lower()
                    results.append(predicted == ground_truth)
                else:
                    results.append(False)

        # Task 19 is path prediction
        elif item["task"] == "qa19":
            ground_truth = ground_truth.split(",")
            for response in responses:
                # Extract only the directions from the response
                pattern = r"\b[nsew]\b"
                directions = re.findall(pattern, response, flags=re.I)
                if not directions:
                    # Try north east south west in full words
                    pattern = r"\b(north|east|south|west)\b"
                    directions = re.findall(pattern, response, flags=re.I)
                    directions = [d[0].lower() for d in directions]
                directions = [d.lower() for d in directions]
                # Check if the predicted directions match the ground truth (in any order)
                results.append(
                    directions == ground_truth or directions[::-1] == ground_truth
                )

        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracy.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracy.
        """
        # First value is the sum of scores, second is the count
        scores = {task: [0.0, 0] for task in ["qa17", "qa18", "qa19"]}
        total_scores = [0.0, 0]
        for item in dataset:
            mean_score = sum(item["scores"]) / len(item["scores"])
            scores[item["task"]][0] += mean_score
            scores[item["task"]][1] += 1
            total_scores[0] += mean_score
            total_scores[1] += 1

        final_scores = {}
        final_scores["total_accuracy"] = (
            total_scores[0] / total_scores[1] if total_scores[1] > 0 else 0.0
        )
        final_scores["total_count"] = total_scores[1]
        final_scores["by_category"] = {
            task: {"accuracy": total / count if count > 0 else 0.0, "count": count}
            for task, (total, count) in scores.items()
        }

        return final_scores
