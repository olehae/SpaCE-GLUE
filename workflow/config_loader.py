"""Configuration loader for SpaCE-GLUE evaluation framework."""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve environment variables in configuration values.
    Supports ${VAR_NAME} syntax for environment variable substitution.
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            load_dotenv()
            resolved = os.environ.get(env_var)
            if resolved is None:
                raise ValueError(
                    f"Environment variable '{env_var}' not found. "
                    f"Please set it before running the workflow."
                )
            return resolved
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _check_instance(name: str, value: Any, expected_type: type):
    """Helper to check type of a value and raise ValueError if mismatched."""
    if not isinstance(value, expected_type):
        raise ValueError(f"'{name}' must be of type {expected_type.__name__}")


def _find_extra_keys(name: str, allowed_keys: set, actual_keys: set):
    """Helper to find unexpected keys in a mapping."""
    extra = actual_keys - allowed_keys
    if extra:
        raise ValueError(
            f"'{name}' contains unsupported keys: {', '.join(sorted(extra))}. "
            f"The only allowed keys are: {', '.join(sorted(allowed_keys))}."
        )


def _validate_and_normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config contents and normalize structure.

    - Ensures required sections exist (`model`, `datasets`).
    - Sets defaults for `evaluation`, `logging`, `inference`, and `scoring`.
    - Performs basic type checks and raises `ValueError` with a helpful message
      if something's incorrect.
    """
    # Define defaults
    EVAL_DEFAULTS = {
        "results_dir": "results",
        "save_every": 50,
        "batch_size": 1,
        "inference": True,
        "scoring": True,
    }
    LOG_DEFAULTS = {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "file": None,
    }

    # required top-level sections
    if "model" not in config:
        raise ValueError("Missing 'model' section in configuration")
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Missing or empty 'datasets' section in configuration")

    # model section
    model = config["model"]
    _check_instance("model", model, dict)
    if "class" not in model or not isinstance(model["class"], str):
        raise ValueError(
            "'model.class' must be provided as a string (full import path)"
        )

    if "params" in model:
        _check_instance("model.params", model["params"], dict)
    else:
        # Ensure a params mapping exists for downstream code
        model["params"] = {}
    # Disallow any other top-level keys besides 'class' and 'params'
    _find_extra_keys("model", {"class", "params"}, set(model.keys()))
    config["model"] = model

    # dataset section
    datasets = config["datasets"]
    _check_instance("datasets", datasets, list)
    normalized_datasets: List[Dict[str, Any]] = []
    for i, entry in enumerate(datasets):
        _check_instance(f"dataset entry at index {i}", entry, dict)

        if "class" not in entry or not isinstance(entry["class"], str):
            raise ValueError(
                f"Dataset entry at index {i} must include 'class' as a string"
            )
        if "params" in entry:
            _check_instance(
                f"dataset entry at index {i} 'params'", entry["params"], dict
            )
        else:
            # Ensure a params mapping exists for downstream code
            entry["params"] = {}
        _find_extra_keys(
            f"dataset entry at index {i}", {"class", "params"}, set(entry.keys())
        )
        normalized_datasets.append(entry)
    config["datasets"] = normalized_datasets

    # evaluation defaults & types
    evaluation = config.get("evaluation", False)
    if evaluation:
        _check_instance("evaluation", evaluation, dict)
        # apply defaults
        for k, v in EVAL_DEFAULTS.items():
            if k in evaluation:
                _check_instance(f"evaluation.{k}", evaluation[k], type(v))
            else:
                evaluation[k] = v
        config["evaluation"] = evaluation
    else:
        config["evaluation"] = EVAL_DEFAULTS

    # logging defaults & types
    logging = config.get("logging", False)
    if logging:
        _check_instance("logging", logging, dict)
        # apply defaults
        for k, v in LOG_DEFAULTS.items():
            # Handle log file separately to allow None
            if k == "file":
                if k in logging and logging[k] is not None:
                    _check_instance(f"logging.{k}", logging[k], str)
                else:
                    logging[k] = v
                continue

            if k in logging:
                _check_instance(f"logging.{k}", logging[k], type(v))
            else:
                logging[k] = v
        config["logging"] = logging
    else:
        config["logging"] = LOG_DEFAULTS

    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Configuration file is empty")

    _check_instance("config", config, dict)

    # Resolve environment variables
    config = _resolve_env_vars(config)

    # Validate and normalize schema; fill defaults
    return _validate_and_normalize_config(config)
