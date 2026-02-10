from space_glue.workflow.runner import run_workflow
import argparse
import sys
import asyncio


def main(argv=None):
    """Console script entry point for running the workflow."""
    parser = argparse.ArgumentParser(description="Run SpaCE-GLUE evaluation workflow")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_workflow(config_path=args.config))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
