import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from rich.progress import track


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
        fname = f"{dataset.name}_{model.name}.jsonl"
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

    def inference(
        self,
        dataset: Iterable[Dict[str, Any]],
        model: Any,
        batch_size: int = 1,
        system_prompt: str = "You are a helpful assistant.",
        runs: int = 1,
    ) -> Dict[str, Any]:
        """Run inference.

        Args:
            dataset: iterable of items (dataset should expose `.name` attribute)
            model: model instance implementing `generate_single` and/or `generate_batch`
            batch_size: how many prompts to send at once (use >1 only if dataset.batch_possible)
            system_prompt: system prompt to pass to the model
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

        # Buffer for batch mode
        batch_items: List[Dict[str, Any]] = []

        with results_path.open("a", encoding="utf-8") as out_file:
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
                    raise RuntimeError(f"index={item['index']}: to_prompt error: {e}")

                # Batch mode
                if getattr(dataset, "batch_possible", False) and batch_size > 1:
                    batch_items.append(item)

                    if len(batch_items) >= batch_size:
                        resps = []
                        for _ in range(runs):
                            try:
                                resp = model.generate_batch(
                                    [dataset.to_prompt(item) for item in batch_items],
                                    system_prompt=system_prompt,
                                )
                                resps.append(resp)
                            except Exception as e:
                                raise RuntimeError(
                                    f"items={batch_items}: generate_batch error: {e!s}"
                                )

                            # TODO: store batch results efficiently
                            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            written += 1  # TODO: adjust written count for batch
                        out_file.flush()
                        batch_items.clear()

                # Single-item mode
                else:
                    resps = []
                    for _ in range(runs):
                        try:
                            resp = model.generate_single(
                                prompt, system_prompt=system_prompt
                            )
                            resps.append(resp)
                        except Exception as e:
                            raise RuntimeError(
                                f"index={item['index']}: generate_single error: {e}"
                            )

                    entry = {
                        "index": item["index"],
                        "responses": resps,
                    }
                    # write the successful single response
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_file.flush()
                    written += 1

        # leftover batch if any
        if batch_items:
            with results_path.open("a", encoding="utf-8") as out_file:
                resps = []
                for _ in range(runs):
                    try:
                        resp = model.generate_batch(
                            [dataset.to_prompt(item) for item in batch_items],
                            system_prompt=system_prompt,
                        )
                        resps.append(resp)
                    except Exception as e:
                        raise RuntimeError(
                            f"items={batch_items}: generate_batch error: {e}"
                        )

                # TODO: store batch results efficiently

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
    ) -> Dict[str, Any]:
        """Score previously generated results using the dataset's evaluate function.

        This method:
        1. Loads all results from the JSONL file
        2. Calls dataset.evaluate() with the full dataset to get per-item scores
        3. Adds scores to each entry in the results file
        4. Calls dataset.aggregate() to compute category-wise statistics
        5. Saves aggregated results to a summary JSON file

        Args:
            dataset: dataset instance with `evaluate(dataset)` and `aggregate(dataset)` methods
            model: model instance (used to locate the results file)

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

        # Call dataset.evaluate() with full dataset to get per-item scores
        try:
            eval_results = dataset.evaluate(dataset_items)
        except Exception as e:
            raise RuntimeError(f"Dataset evaluate error: {e}")

        # eval_results should be a dict mapping index -> score
        if not isinstance(eval_results, dict):
            raise ValueError(
                f"dataset.evaluate() must return a dict, got {type(eval_results)}"
            )

        # Add scores to dataset_items
        for item in dataset_items:
            idx = item["index"]
            if idx not in eval_results:
                raise ValueError(f"Missing score for index {idx} in evaluation results")
            item["scores"] = eval_results[idx]

        # Rewrite the results file with scores included
        with results_path.open("w", encoding="utf-8") as out_file:
            for item in dataset_items:
                out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

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
            "summary_path": str(summary_path),
            "num_scored": len(dataset_items),
            "aggregated_results": aggregated,
        }
