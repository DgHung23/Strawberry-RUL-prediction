from __future__ import annotations

import argparse
from pathlib import Path

from .compare_models import run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for older tuned-model comparison workflow. "
            "For the new strawberry pipeline, it compares a completed LOOCV run."
        )
    )
    parser.add_argument("--run-name", default=None, help="Optional run folder name under output/runs/strawberry.")
    parser.add_argument("--run-root", type=Path, default=None, help="Explicit run root containing fold_results.csv.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Output graph directory.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output report directory.")
    parser.add_argument("--max-points-per-model", type=int, default=2500, help="Scatter sampling cap per model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    if run_root is None and args.run_name:
        run_root = Path("output") / "runs" / "strawberry" / args.run_name
    run_comparison(run_root=run_root, graph_dir=args.graph_dir, report_dir=args.report_dir, max_points_per_model=args.max_points_per_model)


if __name__ == "__main__":
    main()

