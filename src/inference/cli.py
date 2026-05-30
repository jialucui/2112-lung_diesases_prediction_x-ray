"""CLI entry: pass one or more image paths for chest X-ray inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.config_paths import default_checkpoint_path, resolve_config_path
from src.inference.predictor import PneumoniaPredictor


def main() -> None:
    default_cfg = resolve_config_path(ROOT)
    parser = argparse.ArgumentParser(
        description="Chest X-ray classification + severity (assistive, not medical advice)",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths (jpg/png/dcm, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="Config YAML (default: chest deploy config if present)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pth (default: from config paths or LUNG_XRAY_CHECKPOINT)",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument(
        "--no-dataset-norm",
        action="store_true",
        help="Use ImageNet normalization instead of dataset stats",
    )
    parser.add_argument("--age", type=float, default=None, help="Patient age (tabular models)")
    parser.add_argument("--gender", type=str, default=None, help="M / F (tabular models)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = str(default_checkpoint_path(ROOT, config))
    elif not Path(ckpt).is_absolute():
        ckpt = str(ROOT / ckpt)

    predictor = PneumoniaPredictor.from_config_file(
        str(config_path),
        checkpoint_path=ckpt,
        device=args.device,
        use_dataset_normalization=not args.no_dataset_norm,
    )
    inf_cfg = predictor.config.get("inference") or {}
    thr = float(inf_cfg.get("confidence_threshold", 0.8))

    for img in args.images:
        result = predictor.predict(img, age=args.age, gender=args.gender, confidence_threshold=thr)
        if args.json:
            print(PneumoniaPredictor.to_json(result))
        else:
            print(PneumoniaPredictor.format_report(result))
            print()


if __name__ == "__main__":
    main()
