"""Workflow runner for SpaCE-GLUE evaluation framework."""

import sys
import logging
import importlib
from typing import Any

from workflow.config_loader import load_config
from evaluation.evaluator import Evaluator


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})

    logger = logging.getLogger("SpaCE-GLUE")
    logger.setLevel(log_config.get("level", "INFO"))
    logger.handlers.clear()

    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_config.get("file"):
        file_handler = logging.FileHandler(
            log_config["file"], mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_class(module_path: str, class_name: str, **kwargs) -> Any:
    """Dynamically load and instantiate a class.

    Args:
        module_path: Python module path (e.g., "data.my_dataset")
        class_name: Class name to instantiate
        **kwargs: Arguments to pass to class constructor

    Returns:
        Instance of the loaded class
    """
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def run_workflow(
    config_path: str = "config.yaml", inference: bool = True, scoring: bool = True
):
    """Run the evaluation workflow.

    Args:
        config_path: Path to configuration YAML file
        inference: Whether to run inference phase
        scoring: Whether to run scoring phase
    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("Starting SpaCE-GLUE Evaluation Workflow")

    # Setup evaluator
    eval_config = config.get("evaluation", {})
    evaluator = Evaluator(
        results_dir=eval_config.get("results_dir", "results"),
        save_every=eval_config.get("save_every", 50),
    )

    # Load model
    model_config = config["model"]
    logger.info(f"Loading model from {model_config['module']}.{model_config['class']}")

    model_params = {
        k: v for k, v in model_config.items() if k not in ["module", "class"]
    }
    model = load_class(model_config["module"], model_config["class"], **model_params)

    logger.info(f"Model loaded: {model.name}")

    # Load and process datasets
    for dataset_config in config["datasets"]:
        logger.info(
            f"Loading dataset from {dataset_config['module']}.{dataset_config['class']}"
        )

        dataset_params = dataset_config.get("params", {})
        dataset = load_class(
            dataset_config["module"], dataset_config["class"], **dataset_params
        )

        logger.info(f"Dataset loaded: {dataset.name} ({len(dataset)} items)")

        # Inference
        if inference:
            try:
                logger.info(f"Running inference on {dataset.name}")
                summary = evaluator.inference(
                    dataset=dataset,
                    model=model,
                    batch_size=eval_config.get("batch_size", 1),
                    system_prompt=eval_config.get(
                        "system_prompt", "You are a helpful assistant."
                    ),
                )
                logger.info(f"Inference complete: {summary['written']} items")
            except FileExistsError as e:
                logger.warning(f"Skipping inference: {e}")

        # Scoring
        if scoring:
            try:
                logger.info(f"Scoring results for {dataset.name}")
                summary = evaluator.score_results(dataset=dataset, model=model)
                logger.info(f"Scoring complete: {summary['num_scored']} items")

                if (
                    "aggregated_results" in summary
                    and "overall" in summary["aggregated_results"]
                ):
                    logger.info(f"Results: {summary['aggregated_results']['overall']}")
            except FileNotFoundError as e:
                logger.warning(f"No results to score: {e}")

    logger.info("Workflow completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SpaCE-GLUE evaluation workflow")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference phase"
    )
    parser.add_argument(
        "--skip-scoring", action="store_true", help="Skip scoring phase"
    )

    args = parser.parse_args()

    try:
        run_workflow(
            config_path=args.config,
            inference=not args.skip_inference,
            scoring=not args.skip_scoring,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
