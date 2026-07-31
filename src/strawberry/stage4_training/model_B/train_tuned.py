"""Tuned training script for Model B.

This script applies the strawberry lab's first recommended deep-model config:

- strawberry split path: data/03_split/strawberry
- image-dominant late environment branch
- temporal pooling over last + mean + max LSTM states
- light augmentation
- SmoothL1/Huber-style loss
- ReduceLROnPlateau and early stopping

Outputs are written separately from the original Model B run:

- models/model_B_tuned/best_model.pth
- data/model_B_tuned_outputs/training_history.csv
- data/model_B_tuned_outputs/test_predictions.csv
- data/model_B_tuned_outputs/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StrawberrySequenceDataset
from model import StrawberryRULModelB

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_split_root(project_root: Path) -> Path:
    strawberry_root = project_root / "data" / "03_split" / "strawberry"
    if strawberry_root.exists():
        return strawberry_root
    return project_root / "data" / "03_split"


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def make_loader_options(args: argparse.Namespace, device: torch.device) -> dict:
    options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda" and not args.no_pin_memory,
    }
    if args.num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = args.prefetch_factor
    return options


def make_loss(name: str, beta: float) -> nn.Module:
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=beta)
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")


def metric_bundle(preds: list[float], targets: list[float]) -> dict[str, float]:
    pred = torch.tensor(preds, dtype=torch.float32)
    target = torch.tensor(targets, dtype=torch.float32)
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    nonzero = target != 0
    mape = (
        torch.mean(torch.abs((pred[nonzero] - target[nonzero]) / target[nonzero])).item() * 100
        if int(nonzero.sum()) > 0
        else float("nan")
    )
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def optimizer_for_model(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    freeze_backbone: bool,
) -> optim.Optimizer:
    if freeze_backbone:
        return optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("cnn_features") or name.startswith("cnn_pool"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    preds: list[float] = []
    targets: list[float] = []

    desc = "Training" if training else "Evaluating"
    for images, envs, ruls in tqdm(
        loader,
        desc=desc,
        file=sys.stdout,
        mininterval=15,
        miniters=25,
    ):
        images = images.to(device, non_blocking=True)
        envs = envs.to(device, non_blocking=True)
        ruls = ruls.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images, envs)
            loss = criterion(outputs, ruls)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        preds.extend(outputs.detach().cpu().numpy().flatten().tolist())
        targets.extend(ruls.detach().cpu().numpy().flatten().tolist())

    metrics = metric_bundle(preds, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["preds"] = preds
    metrics["targets"] = targets
    return metrics


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[4]
    split_root = resolve_split_root(project_root)
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    run_name = args.run_name.strip().replace(" ", "_")
    if not run_name:
        raise ValueError("--run-name must not be empty")

    models_dir = project_root / "models" / f"model_B_{run_name}"
    outputs_dir = project_root / "data" / f"model_B_{run_name}_outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Split root: {split_root}")
    print(f"AMP enabled: {use_amp}")

    train_transform, eval_transform = build_transforms()
    train_dataset = StrawberrySequenceDataset(train_dir, seq_len=args.seq_len, transform=train_transform)
    val_dataset = StrawberrySequenceDataset(val_dir, seq_len=args.seq_len, transform=eval_transform)
    test_dataset = StrawberrySequenceDataset(test_dir, seq_len=args.seq_len, transform=eval_transform)
    loader_options = make_loader_options(args, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")
    print(f"Test sequences:  {len(test_dataset)}")

    model = StrawberryRULModelB(
        rnn_hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        fusion_mode=args.fusion_mode,
        temporal_pooling=args.temporal_pooling,
        env_hidden_size=args.env_hidden_size,
    ).to(device)

    criterion = make_loss(args.loss, args.smooth_l1_beta)
    optimizer = optimizer_for_model(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    last_checkpoint_path = models_dir / "last_checkpoint.pth"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model B tuned config:")
    print(f"  run_name={run_name}")
    print(f"  fusion_mode={args.fusion_mode}")
    print(f"  temporal_pooling={args.temporal_pooling}")
    print(f"  seq_len={args.seq_len}")
    print(f"  batch_size={args.batch_size}")
    print(f"  loss={args.loss}")
    print(f"  learning_rate={args.learning_rate}")
    print(f"  weight_decay={args.weight_decay}")
    print(f"  dropout={args.dropout}")
    print(f"  num_workers={args.num_workers}")
    print(f"  pin_memory={not args.no_pin_memory}")
    print(f"  freeze_backbone={args.freeze_backbone}")
    print(f"  total_params={total_params:,}")
    print(f"  trainable_params={trainable_params:,}")

    if args.dry_run:
        images, envs, ruls = next(iter(train_loader))
        images = images.to(device)
        envs = envs.to(device)
        with torch.no_grad():
            outputs = model(images, envs)
        print("Dry run batch:")
        print(f"  images:  {tuple(images.shape)}")
        print(f"  envs:    {tuple(envs.shape)}")
        print(f"  ruls:    {tuple(ruls.shape)}")
        print(f"  outputs: {tuple(outputs.shape)}")
        return {"dry_run": True}

    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    start_epoch = 1
    if args.resume:
        if last_checkpoint_path.exists():
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            history = list(checkpoint.get("history", []))
            best_val_mae = float(checkpoint.get("best_val_mae", best_val_mae))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            bad_epochs = int(checkpoint.get("bad_epochs", bad_epochs))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resuming from {last_checkpoint_path} at epoch {start_epoch}/{args.epochs}")
        else:
            print(f"Resume requested but {last_checkpoint_path} was not found; starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
            )

        current_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step(float(val_metrics["mae"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_mae": float(train_metrics["mae"]),
                "val_loss": float(val_metrics["loss"]),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": float(val_metrics["rmse"]),
                "lr": current_lr,
            }
        )

        print(
            "Train MAE: "
            f"{float(train_metrics['mae']):.4f} | "
            f"Val MAE: {float(val_metrics['mae']):.4f} | "
            f"Val RMSE: {float(val_metrics['rmse']):.4f}"
        )

        if float(val_metrics["mae"]) < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            print(f"  >> Saved best checkpoint, val_mae={best_val_mae:.4f}")
        else:
            bad_epochs += 1
            print(f"  No improvement for {bad_epochs}/{args.patience} epoch(s)")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "args": vars(args),
            },
            last_checkpoint_path,
        )

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(outputs_dir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(models_dir / "best_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            use_amp=use_amp,
        )

    predictions = pd.DataFrame(
        {
            "predicted_rul": test_metrics["preds"],
            "actual_rul": test_metrics["targets"],
        }
    )
    predictions.to_csv(outputs_dir / "test_predictions.csv", index=False)

    metrics = {
        "model": f"Model_B_{run_name}_MobileNetV2_CBAM_LSTM",
        "run_name": run_name,
        "mae": float(test_metrics["mae"]),
        "rmse": float(test_metrics["rmse"]),
        "mape": float(test_metrics["mape"]),
        "r2": float(test_metrics["r2"]),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "test_sequences": len(test_dataset),
        "seq_len": args.seq_len,
        "fusion_mode": args.fusion_mode,
        "temporal_pooling": args.temporal_pooling,
        "loss": args.loss,
        "epochs": args.epochs,
        "completed_epochs": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "pin_memory": not args.no_pin_memory,
        "resume": args.resume,
        "smooth_l1_beta": args.smooth_l1_beta,
        "freeze_backbone": args.freeze_backbone,
    }
    with open(outputs_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal test metrics:")
    for key in ["mae", "rmse", "mape", "r2"]:
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Outputs saved to {outputs_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned Model B for strawberry RUL.")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--env-hidden-size", type=int, default=32)
    parser.add_argument(
        "--fusion-mode",
        choices=["early_concat", "late_env_branch", "gated_env_branch"],
        default="late_env_branch",
    )
    parser.add_argument(
        "--temporal-pooling",
        choices=["last", "last_mean_max"],
        default="last_mean_max",
    )
    parser.add_argument("--loss", choices=["mae", "smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--smooth-l1-beta", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--run-name", default="batch128_lr7e4_nopin")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pin-memory", dest="no_pin_memory", action="store_true", default=True)
    parser.add_argument("--pin-memory", dest="no_pin_memory", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
