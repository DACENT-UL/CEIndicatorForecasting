"""Train CE indicator models from a single command.

Examples:
    python -m src.train_models --model ridge
    python -m src.train_models --model all
"""

import argparse
import subprocess
import sys


MODEL_MODULES = {
    "ridge": "src.models.ridge",
    "lasso": "src.models.lasso",
    "rf": "src.models.rf",
    "xgb": "src.models.xgb",
    "svr": "src.models.svr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more model-training modules.")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_MODULES.keys()],
        default="all",
        help="Model to run (default: all)",
    )
    return parser.parse_args()


def run_model(module_path: str) -> int:
    print(f"Running {module_path}...")
    result = subprocess.run([sys.executable, "-m", module_path], check=False)
    return result.returncode


def main() -> int:
    args = parse_args()
    to_run = MODEL_MODULES.items() if args.model == "all" else [(args.model, MODEL_MODULES[args.model])]

    failures = []
    for model_name, module_path in to_run:
        return_code = run_model(module_path)
        if return_code != 0:
            failures.append((model_name, return_code))

    if failures:
        print("\nSome model runs failed:")
        for model_name, code in failures:
            print(f"- {model_name}: exit code {code}")
        return 1

    print("\nAll requested model runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
