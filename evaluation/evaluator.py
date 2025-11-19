import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class Evaluator:
    """Run a dataset through a model, store results, and resume interrupted runs.

    Results are stored as JSON Lines in `results/<dataset.name>_<model.name>.jsonl`.
    Each entry contains at minimum: `index`, `prompt`, and `response`.
    """

    def __init__(self, results_dir: str = "results", save_every: int = 50):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every)

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
    ) -> Dict[str, Any]:
        """Run inference.

        Args:
            dataset: iterable of items (dataset should expose `.name` attribute)
            model: model instance implementing `generate_single` and/or `generate_batch`
            batch_size: how many prompts to send at once (use >1 only if dataset.batch_possible)
            system_prompt: system prompt to pass to the model

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
        batch_indices: List[int] = []
        batch_prompts: List[str] = []
        batch_items: List[Dict[str, Any]] = []

        with results_path.open("a", encoding="utf-8") as out_file:
            for idx, item in enumerate(dataset):
                if idx in done:
                    continue

                try:
                    prompt = dataset.to_prompt(item)
                except Exception as e:
                    raise RuntimeError(f"index={idx}: to_prompt error: {e!s}")

                # Batch mode
                if getattr(dataset, "batch_possible", False) and batch_size > 1:
                    batch_indices.append(idx)
                    batch_prompts.append(prompt)
                    batch_items.append(item)

                    if len(batch_prompts) >= batch_size:
                        try:
                            responses = model.generate_batch(
                                batch_prompts, system_prompt=system_prompt
                            )
                        except TypeError:
                            # some model implementations may have different signature
                            responses = model.generate_batch(batch_prompts)
                        except Exception as e:
                            raise RuntimeError(
                                f"indices={list(batch_indices)}: generate_batch error: {e!s}"
                            )

                        # pair responses and write only successful outputs
                        for bidx, bprompt, bitem, bresp in zip(
                            batch_indices, batch_prompts, batch_items, responses
                        ):
                            entry = {
                                "index": bidx,
                                "prompt": bprompt,
                                "item": bitem,
                                "response": bresp,
                            }
                            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            written += 1
                        out_file.flush()
                        batch_indices.clear()
                        batch_prompts.clear()
                        batch_items.clear()

                # Single-item mode
                else:
                    try:
                        resp = model.generate_single(
                            prompt, system_prompt=system_prompt
                        )
                    except Exception as e:
                        raise RuntimeError(f"index={idx}: generate_single error: {e!s}")

                    entry = {
                        "index": idx,
                        "prompt": prompt,
                        "item": item,
                        "response": resp,
                    }
                    # write the successful single response
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_file.flush()
                    written += 1

        # leftover batch if any
        if batch_prompts:
            with results_path.open("a", encoding="utf-8") as out_file:
                try:
                    responses = model.generate_batch(
                        batch_prompts, system_prompt=system_prompt
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"indices={list(batch_indices)}: generate_batch error: {e!s}"
                    )
                for bidx, bprompt, bitem, bresp in zip(
                    batch_indices, batch_prompts, batch_items, responses
                ):
                    entry = {
                        "index": bidx,
                        "prompt": bprompt,
                        "item": bitem,
                        "response": bresp,
                    }
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
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

                if (
                    "index" not in entry
                    or "item" not in entry
                    or "response" not in entry
                ):
                    raise ValueError(f"Missing required fields in entry: {entry}")

                dataset_items.append(entry)

        if not dataset_items:
            raise ValueError(f"No valid entries found in {results_path}")

        # Call dataset.evaluate() with full dataset to get per-item scores
        try:
            eval_results = dataset.evaluate(dataset_items)
        except Exception as e:
            raise RuntimeError(f"Dataset evaluate error: {e!s}")

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
            item["score"] = eval_results[idx]

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
