#!/usr/bin/env python3
"""Plot training / validation loss curves from log files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reporting import parse_training_log, plot_training_curves


def _merge_chest_logs(histories: list[dict]) -> dict:
    """First run (epochs 1–6) + continuation run (epochs 7–10), avoid duplicate epoch 7."""
    if len(histories) == 1:
        return histories[0]
    h1, h2 = histories[0], histories[1]
    train = list(h1.get("train_loss", []))
    if len(train) >= 6:
        train = train[:6]
    train.extend(h2.get("train_loss", []))
    val_loss = list(h1.get("val_loss", []))
    val_f1 = list(h1.get("val_f1", []))
    # continuation may add val at epoch 10 when training finishes
    if len(h2.get("val_loss", [])) > len(val_loss):
        val_loss.extend(h2.get("val_loss", [])[len(val_loss) :])
        val_f1.extend(h2.get("val_f1", [])[len(val_f1) :])
    return {
        "train_loss": train,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_accuracy_proxy": val_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "logs/train_chest_xray_20ep.log",
            "logs/train_chest_xray_7to10.log",
        ],
    )
    parser.add_argument("--output", default="outputs/chest_xray_report")
    args = parser.parse_args()

    histories = []
    for rel in args.logs:
        p = ROOT / rel
        if p.is_file():
            histories.append(parse_training_log(p))

    if not histories:
        raise SystemExit("No log files found.")

    merged = _merge_chest_logs(histories)
    out = ROOT / args.output
    plot_training_curves(merged, out)

    with open(out / "training_history_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved curves to {out}/")
    print(f"  - train_loss_curve.png")
    print(f"  - val_loss_curve.png")
    print(f"  - train_val_loss_combined.png  (recommended)")


if __name__ == "__main__":
    main()
