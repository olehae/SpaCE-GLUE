from .base_dataset import BaseDataset
from .grasp_evaluator import GRASPEvaluator, parse_steps_from_response
from typing import Dict, List, Iterable, Any
from urllib import request
import json


class GRASP(BaseDataset):
    """
    GRASP Dataset
    """

    name: str = "GRASP"
    data_source: str = "data/grasp_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        r"You are an agent in a grid world. The grid world consists of cells. Each cell may have one unit of energy or no energy at all. Some cells are blocked by obstacles. You cannot move to or through these cells.The goal for you is to collect as much energy as possible and put the collected energy back in the cell where you started. You have 20 steps. For each step, you can choose UP, DOWN, LEFT, RIGHT, TAKE, and DROP. UP allows you to move one cell up in one step. The other movements are similar. You can collect energy from a cell by being in the cell and TAKE the energy from the cell. If there is no energy in the cell, you cannot take any energy from it. You can not move across the boundary of the grid world. You can drop all your energy by DROP. You can use less than 20 steps. Any invalid step will not cause any change in the grid world."
    )

    grid_types = [
        "inner_random_block",
        "inner_upDown_block",
        "inner_leftRight_block",
        "inner_cluster_block",
        "inner_spiral_block",
    ]

    # See https://github.com/jasontangzs0/GRASP for original code and data

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the GRASP dataset from a JSONL file.
        If that fails, it attempts to download it from GitHub.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            base_url = (
                "https://raw.githubusercontent.com/jasontangzs0/GRASP/main/data/grids/"
            )
            all_data = []
            global_index = 0

            for grid_type in self.grid_types:
                filename = f"{grid_type}.jsonl"
                url = base_url + filename

                with request.urlopen(url) as response:
                    content = response.read().decode("utf-8")

                # Parse each line as JSON and reindex
                lines = content.strip().split("\n")
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        # Reindex: file 1 (0-99), file 2 (100-199), etc.
                        data["index"] = global_index
                        all_data.append(data)
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
        return (
            "You are given the following as the representation of the grid world, where A is you, E is energy:\n"
            + item["grid"]
            + "\nGive your sequence of steps as a list. For example: [STEP, STEP, ...]"
        )

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A list of scores (final energy collected at start position).
        """
        results = []
        evaluator = GRASPEvaluator(item["grid"])

        for response in item["responses"]:
            steps = parse_steps_from_response(response)
            result = evaluator.evaluate(steps)

            # result includess final_energy, steps_taken, valid_steps
            results.append(result)

        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean accuracy.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated accuracy.
        """
        overall_energy = 0.0
        overall_steps = 0.0
        for item in dataset:
            scores = item["scores"]
            mean_energy = sum(score["final_energy"] for score in scores) / len(scores)
            mean_steps = sum(score["valid_steps"] for score in scores) / len(scores)
            overall_energy += mean_energy
            overall_steps += mean_steps
        return {
            "accuracy": overall_energy / len(dataset),
            "avg_valid_steps": overall_steps / len(dataset),
        }
