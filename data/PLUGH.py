from .base_dataset import BaseDataset
from .plugh_evaluator import extract_graph, task1, task2, task3_4
from typing import Dict, List, Iterable, Any
from urllib import request
import json


class PLUGH(BaseDataset):
    """
    PLUGH Dataset
    """

    name: str = "PLUGH"
    data_source: str = "data/plugh_data.jsonl"
    batch_possible: bool = True

    # Will be set per item based on task number
    system_prompts = {
        1: "You will be provided with a short fiction text. Your task is to extract mentioned locations and compile a description of locations graph in a graphviz format, undirected, without nodes description, only with edges without labels for directly connected nodes.",
        2: "You will be provided with a short fiction text and a list of location names. Your task is to extract the main character's path as a sequence of visited locations, one by one, each on a new line.",
        3: "You will be provided with a short fiction text and a list of location names. Your task is to extract the shortest path between two given locations (source and target) as a sequence of visited locations starting from the source and ending with the target location, one by one, each on a new line.",
        4: "You will be provided with a short fiction text and a list of location names. Your task is to extract the shortest path between two given locations (source and target) as a sequence of visited locations starting from the source and ending with the target location, one by one, each on a new line.",
    }

    # See https://github.com/altsoph/PLUGH for original code and data

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the PLUGH dataset from a JSONL file.
        If that fails, it attempts to download it from GitHub.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            base_url = "https://raw.githubusercontent.com/altsoph/PLUGH/refs/heads/main/plugh.json"
            all_data = []
            global_index = 0

            with request.urlopen(base_url) as response:
                json_content = json.load(response)
                for item in json_content:
                    # Reindex: text 0 -> question 0, 1, 2, 3. text 1 -> question 4, 5, 6, 7, etc.
                    for task_num in range(1, 5):
                        try:
                            specifics = item[f"task{task_num}"]
                            example = specifics["few_shots"][0]
                        except (KeyError, TypeError):
                            # Task not present in this item or no example available -> Skip
                            continue

                        # Prepare one-shot user input
                        if task_num == 1:
                            one_shot_user = "# Input text\n\n" + example[0]
                        elif task_num == 2:
                            locations = "\n".join(example[1])
                            one_shot_user = (
                                "# Input text\n\n"
                                + example[0]
                                + "\n\n# Locations\n\n"
                                + locations
                            )
                        else:
                            locations = "\n".join(example[1])
                            endpoints = example[2][0] + "\n" + example[2][-1]
                            one_shot_user = (
                                "# Input text\n\n"
                                + example[0]
                                + "\n\n# Locations\n\n"
                                + locations
                                + "\n\n# Source and target\n\n"
                                + endpoints
                            )

                        # Prepare one-shot assistant response
                        if task_num in (2, 3, 4):
                            one_shot_assistant = "\n".join(example[-1])
                        elif task_num == 1:
                            one_shot_assistant = example[-1]

                        line = {
                            "index": global_index,
                            "task_type": task_num,
                            "system_prompt": self.system_prompts[task_num],
                            "text": item["text"],
                            "one_shot": {
                                "user": one_shot_user,
                                "assistant": one_shot_assistant,
                            },
                            "locations": item["locations"],
                            "target": specifics["target"],
                        }
                        if task_num == 3:
                            line["from"] = specifics["from"]
                            line["to"] = specifics["to"]
                        elif task_num == 4:
                            line["from_marker"] = specifics["from_marker"]
                            line["to_marker"] = specifics["to_marker"]
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
        task_num = item["task_type"]

        if task_num == 1:
            prompt = "# Input text\n\n" + item["text"]
        elif task_num == 2:
            locations = "\n".join(item["locations"])
            prompt = (
                "# Input text\n\n" + item["text"] + "\n\n# Locations\n\n" + locations
            )
        elif task_num == 3:
            locations = "\n".join(item["locations"])
            endpoints = item["from"] + "\n" + item["to"]
            prompt = (
                "# Input text\n\n"
                + item["text"]
                + "\n\n# Locations\n\n"
                + locations
                + "\n\n# Source and target\n\n"
                + endpoints
            )
        elif task_num == 4:
            locations = "\n".join(item["locations"])
            endpoints = item["from_marker"] + "\n" + item["to_marker"]
            prompt = (
                "# Input text\n\n"
                + item["text"]
                + "\n\n# Locations\n\n"
                + locations
                + "\n\n# Source and target\n\n"
                + endpoints
            )

        return prompt

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A list of scores.
        """
        results = []
        task_num = item["task_type"]
        responses = []
        for resp in item["responses"]:
            if r"</think>" in resp:
                resp = resp.split(r"</think>")[-1].strip()
            responses.append(resp.lower())

        if task_num == 1:
            for response in responses:
                target_graph = item["target"]
                pred_graph = extract_graph(response)
                f1_nodes, f1_edges = task1(target_graph, pred_graph)
                result = {"f1_nodes": f1_nodes, "f1_edges": f1_edges}
                results.append(result)
        elif task_num == 2:
            for response in responses:
                result = task2(item["target"], response)
                results.append(result)
        else:  # task 3 and 4
            for response in responses:
                result = task3_4(item["target"], response)
                results.append(result)

        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean scores.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated scores.
        """
        task1_f1_nodes = 0.0
        task1_f1_edges = 0.0
        count1 = 0
        task2_ldist, count2 = 0.0, 0
        task3_ldist, count3 = 0.0, 0
        task4_ldist, count4 = 0.0, 0
        total_f1 = 0.0
        total_ldist = 0.0
        count_ldist = 0

        for item in dataset:
            scores = item["scores"]
            if item["task_type"] == 1:
                mean_f1_nodes = sum(score["f1_nodes"] for score in scores) / len(scores)
                mean_f1_edges = sum(score["f1_edges"] for score in scores) / len(scores)
                task1_f1_nodes += mean_f1_nodes
                task1_f1_edges += mean_f1_edges
                total_f1 += (mean_f1_nodes + mean_f1_edges) / 2
                count1 += 1
            elif item["task_type"] == 2:
                mean_ldist = sum(scores) / len(scores)
                task2_ldist += mean_ldist
                count2 += 1
                total_ldist += mean_ldist
                count_ldist += 1
            elif item["task_type"] == 3:
                mean_ldist = sum(scores) / len(scores)
                task3_ldist += mean_ldist
                count3 += 1
                total_ldist += mean_ldist
                count_ldist += 1
            elif item["task_type"] == 4:
                mean_ldist = sum(scores) / len(scores)
                task4_ldist += mean_ldist
                count4 += 1
                total_ldist += mean_ldist
                count_ldist += 1

        return {
            "total_f1": total_f1 / count1 if count1 > 0 else 0.0,
            "total_f1_count": count1,
            "total_normalized_levenshtein": (
                total_ldist / count_ldist if count_ldist > 0 else 0.0
            ),
            "total_normalized_levenshtein_count": count_ldist,
            "by_task": {
                "task1": {
                    "nodes_f1": task1_f1_nodes / count1 if count1 > 0 else 0.0,
                    "edges_f1": task1_f1_edges / count1 if count1 > 0 else 0.0,
                    "count": count1,
                },
                "task2": {
                    "normalized_levenshtein": (
                        task2_ldist / count2 if count2 > 0 else 0.0
                    ),
                    "count": count2,
                },
                "task3": {
                    "normalized_levenshtein": (
                        task3_ldist / count3 if count3 > 0 else 0.0
                    ),
                    "count": count3,
                },
                "task4": {
                    "normalized_levenshtein": (
                        task4_ldist / count4 if count4 > 0 else 0.0
                    ),
                    "count": count4,
                },
            },
        }
