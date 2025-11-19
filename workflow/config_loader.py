"""Configuration loader for SpaCE-GLUE evaluation framework."""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict


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

    # Resolve environment variables
    config = _resolve_env_vars(config)

    # Basic validation
    if "model" not in config:
        raise ValueError("Missing 'model' section in configuration")
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Missing or empty 'datasets' section in configuration")

    # Create results directory
    results_dir = config.get("evaluation", {}).get("results_dir", "results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return config
