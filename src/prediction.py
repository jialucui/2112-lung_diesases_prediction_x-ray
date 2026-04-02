"""
Inference for a single image. Model architecture and preprocessing match training.

Usage (from project root):
  python src/prediction.py 1.jpg
  python -m src.prediction path/to/image.jpg
  python -m src.prediction path/to/image.jpg --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.medical_models import create_model
from src.preprocessing.dicom_xray_loader import get_image_statistics


def _resolve(root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _class_names_from_data_dir(data_path: Path) -> list[str] | None:
    if not data_path.is_dir():
        return None
    dirs = sorted(
        d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(dirs) < 2:
        return None
    return [d.name for d in dirs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pneumonia / chest X-ray classification on one image.")
    parser.add_argument("image_path", type=str, help="Path to JPG/PNG image")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Training config YAML (model + paths + data_dir for class order and normalization).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: paths.checkpoint_dir/best_model.pth from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--no-dataset-stats",
        action="store_true",
        help="Use ImageNet mean/std instead of scanning data.data_dir (faster; may mismatch training).",
    )
    args = parser.parse_args()

    config_path = _resolve(ROOT, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    m = config["model"]
    model = create_model(
        model_type=m["model_type"],
        backbone=m["name"],
        pretrained=m.get("pretrained", True),
        device=device,
        num_classes=m.get("num_classes"),
        severity_classes=m.get("severity_classes"),
    )

    ckpt_path = _resolve(ROOT, args.checkpoint) if args.checkpoint else _resolve(ROOT, Path(config["paths"]["checkpoint_dir"]) / "best_model.pth")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    data_dir = data_cfg.get("data_dir")
    data_path = _resolve(ROOT, data_dir) if data_dir else None

    class_names = _class_names_from_data_dir(data_path) if data_path else None
    if not class_names:
        n = int(m.get("num_classes") or 2)
        class_names = [f"class_{i}" for i in range(n)]

    if args.no_dataset_stats or data_path is None or not data_path.is_dir():
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean, std = get_image_statistics(str(data_path))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )

    image_path = Path(args.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)

    model_type = m.get("model_type", "binary")

    with torch.no_grad():
        outputs = model(batch)

        if model_type == "binary":
            logits = outputs
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            print(f"Image: {image_path}")
            print(f"Predicted class ({pred}): {class_names[pred] if pred < len(class_names) else pred}")
            print(f"Class order (same as training folder scan): {class_names}")
            print(f"Probabilities: {probs.cpu().numpy()}")
        else:
            binary_logits, severity_logits = outputs
            binary_probs = torch.softmax(binary_logits, dim=1)
            severity_probs = torch.softmax(severity_logits, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1).item()
            severity_pred = torch.argmax(severity_probs, dim=1).item()
            print(f"Image: {image_path}")
            print(f"Binary pred: {binary_pred}  probs: {binary_probs.cpu().numpy()}")
            print(f"Severity pred: {severity_pred}  probs: {severity_probs.cpu().numpy()}")


if __name__ == "__main__":
    main()
