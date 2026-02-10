from space_glue.data.base_dataset import BaseDataset
from space_glue.data.stbench_evaluator import calculate_distance, result_parser
from typing import Dict, List, Iterable, Any
import json
from urllib import request
import math


class STBench(BaseDataset):
    """
    STBench Dataset
    """

    name: str = "STBench"
    data_source: str = "data/stbench_data.jsonl"
    batch_possible: bool = True

    system_prompt: str = (
        "you are a helpful text completion assistant. Please continue writing the text entered by the human."
    )

    task_types = {
        "knowledge_comprehension": [
            "administrative_region_determination",
            "poi_category_recognition",
            "poi_identification",
            "urban_region_function_recognition",
        ],
        "spatiotemporal_reasoning": [
            "point_trajectory",
            {
                "point_region": [
                    "point_region_2regions",
                    "point_region_3regions",
                    "point_region_4regions",
                    "point_region_5regions",
                ]
            },
            {
                "trajectory_region": [
                    "trajectory_region_length2",
                    "trajectory_region_length4",
                    "trajectory_region_length6",
                    "trajectory_region_length8",
                    "trajectory_region_length10",
                ]
            },
            {
                "trajectory_identification": [
                    "trajectory_identification_downsampling",
                    "trajectory_identification_spatial_offset",
                    "trajectory_identification_staggered_sampling",
                    "trajectory_identification_temporal_offset",
                ]
            },
        ],
        "accurate_calculation": [
            "direction_determination",
            {
                "navigation": [
                    "navigation_with_weights_5",
                    "navigation_with_weights_6",
                    "navigation_with_weights_7",
                    "navigation_with_weights_8",
                    "navigation_without_weights_8",
                    "navigation_without_weights_9",
                    "navigation_without_weights_10",
                    "navigation_without_weights_11",
                ]
            },
            "trajectory_trajectory",
        ],
        "semantic_reasoning": [
            "road_level_judgment",
            "rush_hour_detection",
            "taxi_occupancy_detection",
        ],
        "downstream_applications": [
            {"flow_prediction": ["inflow_prediction", "outflow_prediction"]},
            "trajectory_prediction",
            {
                "trajectory_anomaly_detection": [
                    "trajectory_anomaly_detection_abnormal",
                    "trajectory_anomaly_detection_normal",
                ]
            },
        ],
    }

    dimensions = [
        "knowledge_comprehension",
        "spatiotemporal_reasoning",
        "accurate_calculation",
        "semantic_reasoning",
        "downstream_applications",
    ]

    # This is needed to distribute samples evenly across tasks
    def distribute(self, task_num, cat_num):
        base = task_num // cat_num
        remainder = task_num % cat_num
        return [base + 1 if i < remainder else base for i in range(cat_num)]

    def load_data(self, **kwargs) -> Iterable[Dict[str, Any]]:
        """
        Tries to load the STBench dataset from a JSONL file.
        If that fails, it attempts to download it from GitHub.
        """
        try:
            with open(self.data_source, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # If the file is not found, load from GitHub and save locally
            # 500 samples in total (100 per dimension), evenly distributed across tasks
            # For every task, every possible file is evenly sampled from
            base_url = "https://raw.githubusercontent.com/LwbXc/STBench/refs/heads/main/datasets/basic"
            all_data = []
            global_index = 0

            for dim in self.dimensions:
                tasks = self.task_types[dim]
                per_task = self.distribute(100, len(tasks))  # 100 samples per dimension
                for task, amount in zip(tasks, per_task):
                    if isinstance(task, str):
                        url = f"{base_url}/{dim}/{task}.jsonl"
                        response = request.urlopen(url).read().decode("utf-8")
                        for item in response.strip().splitlines()[:amount]:
                            data = json.loads(item)
                            line = {
                                "index": global_index,
                                "dimension": dim,
                                "task": task,
                                **data,
                            }
                            all_data.append(line)
                            global_index += 1

                    elif isinstance(task, dict):
                        task_name, files = next(iter(task.items()))
                        per_file = self.distribute(amount, len(files))
                        for file, file_amount in zip(files, per_file):
                            url = f"{base_url}/{dim}/{file}.jsonl"
                            response = request.urlopen(url).read().decode("utf-8")
                            for item in response.strip().splitlines()[:file_amount]:
                                data = json.loads(item)
                                line = {
                                    "index": global_index,
                                    "dimension": dim,
                                    "task": task_name,
                                    **data,
                                }
                                all_data.append(line)
                                global_index += 1

            with open(self.data_source, "w", encoding="utf-8") as f:
                for item in all_data:
                    # Clean up some Answer fields for consistency
                    if item["task"] == "trajectory_region":
                        item["Answer"] = item["Answer"][0]
                    elif item["task"] == "trajectory_trajectory":
                        item["Answer"] = item["Answer"][0]
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
        return item["Question"]

    def evaluate(self, item: Dict[str, Any]) -> List[Any]:
        """
        Evaluates the model's responses against the ground truth answers.

        Args:
            dataset: A list of dataset items with model responses.
        Returns:
            A list of scores.
        """
        task = item["task"]
        ground_truth = item["Answer"]
        parser = result_parser[task]
        answers = [parser(ans) for ans in item["responses"]]
        results = []
        # Trajectory prediction and flow prediction need special handling
        if task == "trajectory_prediction":
            # Model needs to predict a (lon, lat) coordinate
            # AE is absolute error in meters
            for ans in answers:
                try:
                    distance = calculate_distance(ans, ground_truth)
                    if distance >= 100000:  # unrealistic large error
                        distance = None
                except Exception:
                    distance = None
                results.append({"ae": distance})
        elif task == "flow_prediction":
            # Model needs to predict a sequence of 6 numbers
            # MAE is mean absolute error over the sequence
            # RMSE is root mean squared error over the sequence
            n = len(ground_truth)
            for ans in answers:
                if ans is None:
                    mae = None
                    rmse = None
                else:
                    mae = sum(abs(gt - a) for gt, a in zip(ground_truth, ans)) / n
                    rmse = math.sqrt(
                        sum((gt - a) ** 2 for gt, a in zip(ground_truth, ans)) / n
                    )
                results.append({"mae": mae, "rmse": rmse})
        else:  # All other tasks use exact match
            for ans in answers:
                results.append(ans == str(ground_truth))

        return results

    def aggregate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates evaluation results to compute overall mean scores.

        Args:
            dataset: A list of dataset items with evaluation results.
        Returns:
            The final aggregated scores.
        """
        if not dataset:
            raise ValueError("Dataset is empty. Cannot aggregate results.")
        by_task = {}
        seen = set()
        tasks = [
            x["task"]
            for x in dataset
            if x["task"] not in seen and not seen.add(x["task"])
        ]
        dims = self.dimensions.copy()
        dims.remove("downstream_applications")
        by_category = {cat: [0.0, 0] for cat in dims}  # sum, count
        downstream = [0.0, 0, 0.0, 0]  # accuracy sum, count, mae sum, mae count
        total_accuracy = []
        non_accuracy_count = 0

        for task in tasks:
            if task == "trajectory_prediction":
                mae_sum = 0.0
                rmse_sum = 0.0
                count = 0
                for item in dataset:
                    if item["task"] != task:
                        continue
                    # MAE by averaging absolute errors
                    aes = [e["ae"] for e in item["scores"] if e["ae"] is not None]
                    if not aes:
                        continue
                    mean_score = sum(aes) / len(aes)
                    mae_sum += mean_score
                    # RMSE by averaging squared errors then taking root
                    mean_score = math.sqrt(sum(e**2 for e in aes) / len(aes))
                    rmse_sum += mean_score
                    count += 1
                by_task[task] = {
                    "MAE": mae_sum / count if count > 0 else None,
                    "RMSE": rmse_sum / count if count > 0 else None,
                    "count": count,
                }
                non_accuracy_count += count
                downstream[2] += mae_sum
                downstream[3] += count
            elif task == "flow_prediction":
                mae_sum = 0.0
                rmse_sum = 0.0
                count = 0
                for item in dataset:
                    if item["task"] != task:
                        continue
                    # MAE by averaging maes
                    maes = [e["mae"] for e in item["scores"] if e["mae"] is not None]
                    if not maes:
                        continue
                    mean_score = sum(maes) / len(maes)
                    mae_sum += mean_score
                    # RMSE by averaging rmses
                    rsmes = [e["rmse"] for e in item["scores"] if e["rmse"] is not None]
                    mean_score = math.sqrt(sum(e**2 for e in rsmes) / len(rsmes))
                    rmse_sum += mean_score
                    count += 1
                by_task[task] = {
                    "MAE": mae_sum / count if count > 0 else None,
                    "RMSE": rmse_sum / count if count > 0 else None,
                    "count": count,
                }
                non_accuracy_count += count
                downstream[2] += mae_sum
                downstream[3] += count
            else:  # All other tasks use exact match accuracy
                sum_of_scores = 0.0
                count = 0
                for item in dataset:
                    if item["task"] != task:
                        continue
                    mean_score = sum(item["scores"]) / len(item["scores"])
                    sum_of_scores += mean_score
                    count += 1
                    total_accuracy.append(mean_score)
                    try:
                        by_category[item["dimension"]][0] += mean_score
                        by_category[item["dimension"]][1] += 1
                    except KeyError:
                        downstream[0] += mean_score
                        downstream[1] += 1
                by_task[task] = {
                    "accuracy": sum_of_scores / count if count > 0 else 0.0,
                    "count": count,
                }

        l = len(total_accuracy)
        acc = sum(total_accuracy) / l if l > 0 else 0.0
        std = (
            (sum((s - acc) ** 2 for s in total_accuracy) / (l - 1)) ** 0.5
            if l > 1
            else 0.0
        )
        se = std / (l**0.5) if l > 0 else 0.0

        final_output = {}
        final_output["total"] = {
            "accuracy": acc,
            "standard_deviation": std,
            "standard_error": se,
            "count": l,
            "non_accuracy_count": non_accuracy_count,
        }

        final_output["by_category"] = {
            cat: {
                "accuracy": (items[0] / items[1] if items[1] > 0 else 0.0),
                "count": items[1],
            }
            for cat, items in by_category.items()
            if len(items) == 2
        }
        final_output["by_category"]["downstream_applications"] = {
            "accuracy": (downstream[0] / downstream[1] if downstream[1] > 0 else 0.0),
            "mae": (downstream[2] / downstream[3] if downstream[3] > 0 else 0.0),
            "accuracy_count": downstream[1],
            "mae_count": downstream[3],
        }
        final_output["by_task"] = by_task

        return final_output
