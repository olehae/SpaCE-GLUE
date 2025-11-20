"""CLI for running the SpaCE-GLUE evaluation workflow."""

import argparse
from workflow.runner import run_workflow
import sys

parser = argparse.ArgumentParser(description="Run SpaCE-GLUE evaluation workflow")
parser.add_argument("--config", default="config.yaml", help="Path to config file")
args = parser.parse_args()

try:
    run_workflow(config_path=args.config)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
