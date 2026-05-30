#!/usr/bin/env python3
"""
Generate metrics, curves, error analysis, and demo artifacts (result set B).

Writes outputs/chest_xray_report/ using the current eval pipeline.
Training-time test metrics remain in checkpoints/.../test_metrics.yaml (set A).
See docs/EVALUATION_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reporting import generate_full_report
from src.inference.predictor import PneumoniaPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Full ML evaluation report")
    parser.add_argument("--config", default="src/configs/config_chest_xray.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chest_xray_stopped_epoch7/best_model.pth",
    )
    parser.add_argument("--output", default="outputs/chest_xray_report")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--training-log", default="logs/train_chest_xray_20ep.log")
    parser.add_argument("--demo-image", default=None, help="Optional sample image for demo JSON")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = ROOT / args.output
    log_path = ROOT / args.training_log if args.training_log else None
    summary = generate_full_report(
        config_path=ROOT / args.config,
        checkpoint_path=ROOT / args.checkpoint,
        output_dir=out,
        split=args.split,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        training_log=log_path,
    )
    print(json.dumps(summary, indent=2))

    demo_dir = out / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_img = args.demo_image
    if not demo_img:
        candidates = list((ROOT / "Chest_xray_data1/chest_xray/test/NORMAL").glob("*.jpeg"))[:1]
        if candidates:
            demo_img = str(candidates[0])
    if demo_img and Path(demo_img).is_file():
        predictor = PneumoniaPredictor.from_config_file(
            ROOT / args.config,
            checkpoint_path=ROOT / args.checkpoint,
            device=args.device,
        )
        thr = float(
            predictor.config.get("inference", {}).get("confidence_threshold", args.confidence_threshold)
        )
        result = predictor.predict(demo_img, confidence_threshold=thr)
        result_en = PneumoniaPredictor.result_for_english_ui(result)
        shutil.copy(demo_img, demo_dir / "sample_input.jpg")
        with open(demo_dir / "sample_prediction.json", "w", encoding="utf-8") as f:
            json.dump(result_en, f, indent=2, ensure_ascii=False)
        print(f"Demo saved under {demo_dir}")


if __name__ == "__main__":
    main()
