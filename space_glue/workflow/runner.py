"""Workflow runner for SpaCE-GLUE evaluation framework."""

import sys
import logging
import importlib
from typing import Any

from space_glue.workflow.config_loader import load_config
from space_glue.evaluation.evaluator import Evaluator


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


def load_class(class_path: str, **kwargs) -> Any:
    """Dynamically load and instantiate a class.

    Args:
        class_path: Full class path including module (e.g., "data.my_dataset.MyDataset")
        **kwargs: Arguments to pass to class constructor

    Returns:
        Instance of the loaded class
    """
    module = importlib.import_module(".".join(class_path.split(".")[:-1]))
    cls = getattr(module, class_path.split(".")[-1])
    return cls(**kwargs)


async def run_workflow(config_path: str = "config.yaml"):
    """Run the evaluation workflow.

    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting SpaCE-GLUE Evaluation Workflow")

    # Setup evaluator
    eval_config = config.get("evaluation", {})
    evaluator = Evaluator(
        results_dir=eval_config.get("results_dir", "results"),
    )
    inference = eval_config.get("inference", True)
    evaluate = eval_config.get("evaluate", True)
    aggregate = eval_config.get("aggregate", True)

    # Load model
    model_config = config["model"]
    logger.info(f"Loading model from {model_config['class']}")

    model_params = model_config.get("params", {})
    model = load_class(model_config["class"], **model_params)

    logger.info(f"Model loaded: {model.name}")

    # Load and process datasets
    for i, dataset_config in enumerate(config["datasets"]):
        try:
            logger.info(f"Loading dataset from {dataset_config['class']}")

            dataset_params = dataset_config.get("params", {})
            dataset = load_class(dataset_config["class"], **dataset_params)

            logger.info(f"Dataset loaded: {dataset.name} ({len(dataset)} items)")

            # Inference
            if inference:
                logger.info(f"Running inference on {dataset.name}")
                summary = await evaluator.inference(
                    dataset=dataset,
                    model=model,
                    batch_size=eval_config.get("batch_size", 1),
                    system_prompt=dataset.system_prompt,
                    runs=eval_config["runs"][i],
                )
                logger.info(
                    f"Inference complete: {summary['written']} items written to {summary['results_path']}"
                )

            # Evaluation and/or Aggregation
            if evaluate or aggregate:
                logger.info(
                    f"Scoring results for {dataset.name} (evaluate={evaluate}, aggregate={aggregate})"
                )
                summary = evaluator.score_results(
                    dataset=dataset, model=model, evaluate=evaluate, aggregate=aggregate
                )
                logger.info(
                    f"Scoring complete: {summary['num_scored']} items in {summary['results_path']}"
                )
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_config['class']}: {e}")
            logger.info(f"Skipping dataset and continuing with next one")
            continue

    logger.info("SpaCE-GLUE Workflow completed")
