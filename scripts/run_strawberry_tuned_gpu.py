"""Run tuned strawberry model training with the current Python interpreter.

This wrapper is intentionally small: it launches the four tuned train scripts
sequentially, captures each model's stdout/stderr to a workspace log file, and
writes a JSON status file that can be polled while training runs in the
background.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "data" / "strawberry_tuned_gpu_status.json"

MODEL_SCRIPTS = {
    "A": PROJECT_ROOT / "src" / "strawberry" / "stage4_training" / "model_A" / "train_tuned.py",
    "B": PROJECT_ROOT / "src" / "strawberry" / "stage4_training" / "model_B" / "train_tuned.py",
    "C": PROJECT_ROOT / "src" / "strawberry" / "stage4_training" / "model_C" / "train_tuned.py",
    "D": PROJECT_ROOT / "src" / "strawberry" / "stage4_training" / "model_D" / "train_tuned.py",
}


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(**updates) -> None:
    status = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    status.update(updates)
    status["updated_at"] = timestamp()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def output_dir_for(model_key: str, run_name: str) -> Path:
    return PROJECT_ROOT / "data" / f"model_{model_key}_{run_name}_outputs"


def run_model(model_key: str, args: argparse.Namespace) -> int:
    script = MODEL_SCRIPTS[model_key]
    out_dir = output_dir_for(model_key, args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"train_{args.epochs}epoch.log"

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--env-hidden-size",
        str(args.env_hidden_size),
        "--fusion-mode",
        args.fusion_mode,
        "--temporal-pooling",
        args.temporal_pooling,
        "--loss",
        args.loss,
        "--smooth-l1-beta",
        str(args.smooth_l1_beta),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
    ]
    if args.no_amp:
        cmd.append("--no-amp")
    if args.no_pin_memory:
        cmd.append("--no-pin-memory")
    if args.unfreeze_backbone:
        cmd.append("--unfreeze-backbone")
    if args.resume:
        cmd.append("--resume")

    write_status(
        state="running",
        current_model=model_key,
        current_log=str(log_path.relative_to(PROJECT_ROOT)),
        started_model_at=timestamp(),
    )

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"Started: {timestamp()}\n")
        log.write(f"Interpreter: {sys.executable}\n")
        log.write(f"Command: {subprocess.list2cmdline(cmd)}\n\n")
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\nFinished: {timestamp()}\n")
        log.write(f"Return code: {result.returncode}\n")

    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tuned strawberry GPU training.")
    parser.add_argument("--models", nargs="+", default=["A", "B", "C", "D"], choices=sorted(MODEL_SCRIPTS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--env-hidden-size", type=int, default=32)
    parser.add_argument("--fusion-mode", default="late_env_branch")
    parser.add_argument("--temporal-pooling", default="last_mean_max")
    parser.add_argument("--loss", default="smooth_l1")
    parser.add_argument("--smooth-l1-beta", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--run-name", default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sleep-test", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sleep_test:
        write_status(
            state="sleep_test",
            pid=__import__("os").getpid(),
            started_at=timestamp(),
            sleep_seconds=args.sleep_seconds,
        )
        time.sleep(args.sleep_seconds)
        write_status(state="sleep_test_done", completed_at=timestamp())
        return 0

    write_status(
        state="started",
        pid=__import__("os").getpid(),
        interpreter=sys.executable,
        models=args.models,
        run_name=args.run_name,
        config={
            "epochs": args.epochs,
            "patience": args.patience,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "fusion_mode": args.fusion_mode,
            "temporal_pooling": args.temporal_pooling,
            "loss": args.loss,
            "smooth_l1_beta": args.smooth_l1_beta,
            "unfreeze_backbone": args.unfreeze_backbone,
            "resume": args.resume,
            "no_pin_memory": args.no_pin_memory,
            "amp": not args.no_amp,
        },
        completed=[],
        failed=[],
        started_at=timestamp(),
    )

    completed: list[str] = []
    failed: list[str] = []
    for model_key in args.models:
        rc = run_model(model_key, args)
        if rc == 0:
            completed.append(model_key)
            write_status(state="model_completed", current_model=None, completed=completed, failed=failed)
        else:
            failed.append(model_key)
            write_status(state="failed", current_model=model_key, completed=completed, failed=failed, return_code=rc)
            return rc

    write_status(state="completed", current_model=None, completed=completed, failed=failed, completed_at=timestamp())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
