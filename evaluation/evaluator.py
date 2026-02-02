import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from rich.progress import track
import asyncio


class Evaluator:
    """Run a dataset through a model, store results, and resume interrupted runs.

    Results are stored as JSON Lines in `results/<dataset.name>_<model.name>.jsonl`.
    Each entry contains at minimum: `index`, `prompt`, and `responses` (a list).
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    def _result_path(self, dataset, model) -> Path:
        dataset_name = self._sanitize_name(dataset.name)
        model_name = self._sanitize_name(model.name)
        fname = f"{dataset_name}_{model_name}.jsonl"
        return self.results_dir / fname

    def _load_done_indices(self, path: Path) -> set:
        if not path.exists():
            return set()
        done = set()
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if "index" in obj:
                            done.add(int(obj["index"]))
                    except Exception:
                        # skip malformed lines
                        continue
        except Exception:
            return set()
        return done

    async def inference(
        self,
        dataset: Iterable[Dict[str, Any]],
        model: Any,
        batch_size: int = 1,
        system_prompt: str = None,
        runs: int = 1,
    ) -> Dict[str, Any]:
        """Run inference.

        Args:
            dataset: iterable of items (dataset should expose `.name` attribute)
            model: model instance implementing `generate_single` and/or `generate_batch`
            batch_size: how many prompts to send at once (use >1 only if dataset.batch_possible)
            system_prompt: system prompt to pass to the model, unless overridden per-item
            runs: number of times to run each item through the model

        Returns:
            summary dict with stats and path to results file
        """

        results_path = self._result_path(dataset, model)
        summary_path = results_path.with_suffix(".summary.json")

        # Prevent running inference if there's already an evaluated summary
        if summary_path.exists():
            raise FileExistsError(
                f"Summary file already exists at {summary_path}. "
                "This indicates previous evaluation results exist. "
                "Remove or rename previous evaluation results to continue."
            )

        done = self._load_done_indices(results_path)

        try:
            dataset_length = len(dataset)
        except Exception:
            dataset_length = None
        written = 0

        system_prompt = (
            system_prompt
            if system_prompt is not None
            else "You are a helpful assistant."
        )

        # Check if model and dataset support batch mode and it's requested
        batch_possible = getattr(dataset, "batch_possible", False) and batch_size > 1

        with results_path.open("a", encoding="utf-8") as out_file:
            # Single-item mode
            if not batch_possible:
                for item in track(
                    dataset,
                    total=dataset_length,
                    description=f"Running inference for {dataset.name}",
                ):
                    # All runs completed for this item
                    if item["index"] in done:
                        continue

                    try:
                        prompt = dataset.to_prompt(item)
                    except Exception as e:
                        raise RuntimeError(
                            f"index={item['index']}: to_prompt error: {e}"
                        )

                    # Get system prompt for this item (custom or global)
                    item_system_prompt = item.get("system_prompt", system_prompt)

                    # Get one-shot example if present
                    one_shot = item.get("one_shot", None)

                    # Get answer options if present
                    answer_options = item.get("options", None)

                    resps = []
                    for _ in range(runs):
                        try:
                            resp = await model.generate_single(
                                user_prompt=prompt,
                                system_prompt=item_system_prompt,
                                one_shot=one_shot,
                                answer_options=answer_options,
                            )
                            resps.append(resp)
                        except Exception as e:
                            raise RuntimeError(
                                f"index={item['index']}: generate_single error: {e}"
                            )

                    # Copy all original item data and add responses
                    entry = {**item, "responses": resps}
                    # write the successful single response
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_file.flush()
                    written += 1

            # Batch mode
            else:
                tasks = []
                items = []
                # give batch_size * runs to the model at once
                for item in track(
                    dataset,
                    total=dataset_length,
                    description=f"Running inference for {dataset.name} (batch mode)",
                ):
                    # All runs completed for this item
                    if item["index"] in done:
                        continue

                    try:
                        prompt = dataset.to_prompt(item)
                    except Exception as e:
                        raise RuntimeError(
                            f"index={item['index']}: to_prompt error: {e}"
                        )

                    # Get system prompt for this item (custom or global)
                    item_system_prompt = item.get("system_prompt", system_prompt)

                    # Get one-shot example if present
                    one_shot = item.get("one_shot", None)

                    # Get answer options if present
                    answer_options = item.get("options", None)

                    tasks.extend(
                        [
                            {
                                "user_prompt": prompt,
                                "system_prompt": item_system_prompt,
                                "one_shot": one_shot,
                                "answer_options": answer_options,
                            }
                        ]
                        * runs
                    )

                    items.append(item)

                    if len(tasks) >= batch_size * runs:
                        coros = [model.generate_single(**task) for task in tasks]
                        results = await asyncio.gather(*coros)
                        for i, b_item in enumerate(items):
                            item_resps = results[i * runs : (i + 1) * runs]
                            # Copy all original item data and add responses
                            entry = {**b_item, "responses": item_resps}
                            # write the successful single response
                            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            out_file.flush()
                            written += 1
                        tasks = []
                        items = []
                # process any remaining items
                if tasks:
                    coros = [model.generate_single(**task) for task in tasks]
                    results = await asyncio.gather(*coros)
                    for i, b_item in enumerate(items):
                        item_resps = results[i * runs : (i + 1) * runs]
                        # Copy all original item data and add responses
                        entry = {**b_item, "responses": item_resps}
                        # write the successful single response
                        out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        out_file.flush()
                        written += 1

        summary = {
            "results_path": str(results_path),
            "written": written,
            "dataset_length": dataset_length,
        }
        return summary

    def score_results(
        self,
        dataset: Any,
        model: Any,
        evaluate: bool = True,
        aggregate: bool = True,
    ) -> Dict[str, Any]:
        """Score previously generated results using the dataset's evaluate function.

        This method:
        1. Loads all results from the JSONL file
        ### If evaluate=True:
        2. Calls dataset.evaluate() with the full dataset to get per-item scores
        3. Adds scores to each entry in the results file
        ### If aggregate=True:
        4. Calls dataset.aggregate() to compute category-wise statistics
        5. Saves aggregated results to a summary JSON file

        Args:
            dataset: dataset instance with `evaluate(dataset)` and `aggregate(dataset)` methods
            model: model instance (used to locate the results file)
            aggregate: whether to compute and save aggregated results

        Returns:
            dict with aggregated scores and paths to results files
        """
        results_path = self._result_path(dataset, model)

        if not results_path.exists():
            raise FileNotFoundError(f"No results file found at {results_path}")

        # Load all entries from results file
        dataset_items: List[Dict[str, Any]] = []

        with results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON line: {e!s}")

                if "index" not in entry or "responses" not in entry:
                    raise ValueError(f"Missing required fields in entry: {entry}")

                dataset_items.append(entry)

        if not dataset_items:
            raise ValueError(f"No valid entries found in {results_path}")

        if evaluate:
            # Add scores to dataset_items
            for item in dataset_items:
                try:
                    item["scores"] = dataset.evaluate(item)
                except Exception as e:
                    raise RuntimeError(
                        f"Dataset evaluate error for item {item['index']}: {e}"
                    )

            # Rewrite the results file with scores included
            with results_path.open("w", encoding="utf-8") as out_file:
                for item in dataset_items:
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

        if aggregate:
            # Call dataset.aggregate() to compute category-wise statistics
            try:
                aggregated = dataset.aggregate(dataset_items)
            except Exception as e:
                raise RuntimeError(f"Dataset aggregate error: {e!s}")

            # Save aggregated results to summary file
            summary_path = results_path.with_suffix(".summary.json")
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(aggregated, f, indent=2, ensure_ascii=False)

        return {
            "results_path": str(results_path),
            "num_scored": len(dataset_items),
            "summary_path": str(summary_path) if aggregate else None,
            "aggregated_results": aggregated if aggregate else None,
        }
