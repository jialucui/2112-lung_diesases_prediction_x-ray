#!/usr/bin/env python3
"""Save a Grad-CAM overlay PNG for a single chest X-ray image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from src.inference.predictor import PneumoniaPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for one image")
    parser.add_argument("--image", required=True, help="Path to JPG/PNG X-ray")
    parser.add_argument("--config", default="src/configs/config_chest_xray.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/chest_xray_stopped_epoch7/best_model.pth",
    )
    parser.add_argument("--output", default="outputs/grad_cam_overlay.png")
    parser.add_argument("--class-index", type=int, default=None, help="Target class index (default: argmax)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    predictor = PneumoniaPredictor.from_config_file(
        ROOT / args.config,
        checkpoint_path=ROOT / args.checkpoint,
        device=args.device,
    )
    pred = predictor.predict(image_path)
    gc = predictor.grad_cam(
        image_path,
        target_class_index=args.class_index if args.class_index is not None else pred.get("predicted_class_index"),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_url = gc["grad_cam_image"]
    b64 = data_url.split(",", 1)[1]
    import base64

    out.write_bytes(base64.b64decode(b64))
    print(f"Prediction: {pred.get('predicted_class')} (conf {pred.get('confidence_score')})")
    print(f"Grad-CAM target: {gc.get('target_class')}")
    print(f"Saved overlay: {out.resolve()}")


if __name__ == "__main__":
    main()
