#!/usr/bin/env python3
"""Regenerate confusion matrix PNG from metrics YAML or full test-set eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reporting import generate_full_report, plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrix with large labels")
    parser.add_argument("--config", default="src/configs/config_chest_xray.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chest_xray_stopped_epoch7/best_model.pth",
    )
    parser.add_argument("--output", default="outputs/chest_xray_report")
    parser.add_argument(
        "--metrics-yaml",
        default=None,
        help="Use existing metrics YAML instead of re-running inference",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reeval", action="store_true", help="Re-run test inference (slower)")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix.png"

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names = (cfg.get("inference") or {}).get("class_display_names") or [
        f"class_{i}" for i in range(int(cfg["model"]["num_classes"]))
    ]

    if args.reeval:
        generate_full_report(
            config_path=ROOT / args.config,
            checkpoint_path=ROOT / args.checkpoint,
            output_dir=out_dir,
            split=args.split,
            device=args.device,
        )
        print(f"Full report written under {out_dir}")
        return

    metrics_path = Path(args.metrics_yaml) if args.metrics_yaml else out_dir / f"{args.split}_metrics.yaml"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path} (use --reeval to compute)")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = yaml.safe_load(f)
    cm = np.array(metrics["confusion_matrix"], dtype=int)
    plot_confusion_matrix(cm, class_names, out_path)
    print(f"Saved {out_path}  (classes: {class_names}, matrix:\n{cm})")


if __name__ == "__main__":
    main()
