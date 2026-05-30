#!/usr/bin/env python3
"""
Plot ROC curve (AUC) from test_predictions.jsonl or by re-running evaluation.

Default: outputs/chest_xray_report/ (result set B, same checkpoint as training).
AUC 0.9773 matches CM 197/37/13/377, not training-time 149/85/2/388 (AUC 0.9569).
See docs/EVALUATION_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import auc, roc_auc_score, roc_curve

from src.evaluation.reporting import plot_roc


def _load_from_jsonl(jsonl_path: Path, positive_class: str = "PNEUMONIA") -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_score: list[float] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            y_true.append(int(rec["true_label"]))
            probs = rec.get("class_probabilities") or {}
            if positive_class in probs:
                y_score.append(float(probs[positive_class]))
            elif len(probs) == 2:
                # binary: use probability of label 1
                keys = sorted(probs.keys())
                y_score.append(float(probs[keys[1]]))
            else:
                raise ValueError(f"No score for positive class {positive_class!r} in {rec}")
    return y_true, y_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC curve with AUC")
    parser.add_argument(
        "--predictions",
        default="outputs/chest_xray_report/test_predictions.jsonl",
        help="JSONL from generate_ml_report.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/chest_xray_report/roc_curve.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--positive-class",
        default="PNEUMONIA",
        help="Class treated as positive (label 1) for ROC",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Re-run test inference instead of using JSONL",
    )
    parser.add_argument("--config", default="src/configs/config_chest_xray.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chest_xray_stopped_epoch7/best_model.pth",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reeval:
        from src.evaluation.reporting import generate_full_report

        report_dir = out_path.parent
        summary = generate_full_report(
            config_path=ROOT / args.config,
            checkpoint_path=ROOT / args.checkpoint,
            output_dir=report_dir,
            split="test",
            device=args.device,
        )
        auc_val = float(summary.get("metrics", {}).get("auc_roc", 0))
        print(f"Re-eval done. AUC-ROC: {auc_val:.4f}")
        print(f"Saved {report_dir / 'roc_curve.png'}")
        return

    pred_path = ROOT / args.predictions
    if not pred_path.is_file():
        raise SystemExit(
            f"Predictions file not found: {pred_path}\n"
            "Run: python scripts/generate_ml_report.py\n"
            "Or: python scripts/plot_roc_curve.py --reeval"
        )

    y_true, y_score = _load_from_jsonl(pred_path, positive_class=args.positive_class)
    auc_val = float(roc_auc_score(y_true, y_score))
    plot_roc(y_true, y_score, out_path, auc_val)
    print(f"Samples: {len(y_true)}  AUC-ROC: {auc_val:.4f}")
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
