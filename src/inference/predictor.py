"""
Unified inference: 肺炎类型（多分类）+ 严重程度（分档概率与综合百分比估计）.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, p: Union[str, Path]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _class_folder_order(data_path: Path) -> Optional[List[str]]:
    if not data_path.is_dir():
        return None
    dirs = sorted(d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(dirs) < 2:
        return None
    return [d.name for d in dirs]


class PneumoniaPredictor:
    """
    加载与训练一致的 config + checkpoint，对单张/批量图像推理。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        device: str,
        checkpoint_path: Optional[Path] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        m = config["model"]
        self.model_type = m.get("model_type", "binary")
        self.num_classes = int(m.get("num_classes", 2))
        self.severity_classes = int(m.get("severity_classes", 5))

        data_dir = config.get("data", {}).get("data_dir")
        data_path = _resolve_path(PROJECT_ROOT, data_dir) if data_dir else None

        inf = config.get("inference") or {}
        folder_names = _class_folder_order(data_path) if data_path else None
        display = inf.get("class_display_names")
        if display and len(display) == self.num_classes:
            self.class_names: List[str] = list(display)
        elif folder_names and len(folder_names) == self.num_classes:
            self.class_names = folder_names
        else:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

        centers = inf.get("severity_bin_centers")
        if centers and len(centers) == self.severity_classes:
            self.severity_bin_centers = np.array(centers, dtype=np.float32)
        else:
            # 均匀分布在 10%–90%
            self.severity_bin_centers = np.linspace(10.0, 90.0, self.severity_classes).astype(np.float32)

        image_size = int(config.get("data", {}).get("image_size", 224))
        if mean is None or std is None:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        )

        if checkpoint_path and checkpoint_path.is_file():
            ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            try:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            except Exception as e:
                raise RuntimeError(
                    "无法加载权重。若你曾用「仅分类」模型训练，而当前 config 为 multi_task，"
                    "请重新训练生成 checkpoints/best_model.pth，或把 model.model_type 改回与权重一致。"
                ) from e
            logger.info("Loaded checkpoint %s", checkpoint_path)

    @classmethod
    def from_config_file(
        cls,
        config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_dataset_normalization: bool = True,
    ) -> "PneumoniaPredictor":
        from src.models.medical_models import BinaryClassifier, create_model
        from src.preprocessing.dicom_xray_loader import get_image_statistics

        config_path = _resolve_path(PROJECT_ROOT, config_path)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = config["model"]

        ck = config.get("paths", {}).get("checkpoint_dir", "checkpoints/")
        ck_path = _resolve_path(PROJECT_ROOT, Path(ck) / "best_model.pth")
        if checkpoint_path:
            ck_path = _resolve_path(PROJECT_ROOT, checkpoint_path)

        mean = std = None
        data_dir = config.get("data", {}).get("data_dir")
        if use_dataset_normalization and data_dir:
            dp = _resolve_path(PROJECT_ROOT, data_dir)
            if dp.is_dir():
                mean, std = get_image_statistics(str(dp))

        model = create_model(
            model_type=m["model_type"],
            backbone=m["name"],
            pretrained=m.get("pretrained", True),
            device=device,
            num_classes=m.get("num_classes"),
            severity_classes=m.get("severity_classes"),
        )

        if not ck_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

        ckpt = torch.load(str(ck_path), map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            # 旧权重：仅 DenseNet 单头分类（BinaryClassifier，键为 model.*）
            if any(k.startswith("model.") for k in sd) and not any(
                k.startswith("backbone.") for k in sd
            ):
                logger.warning(
                    "检测到仅分类权重（BinaryClassifier）。将忽略 config 中的 multi_task，"
                    "严重程度仅作粗略估计；完整分档请用 multi_task 重新训练。"
                )
                w = sd.get("model.classifier.weight")
                if w is None:
                    raise
                n_cls = int(w.shape[0])
                model = BinaryClassifier(
                    backbone=m["name"],
                    pretrained=False,
                    num_classes=n_cls,
                ).to(device)
                model.load_state_dict(sd, strict=True)
                config = dict(config)
                config["model"] = dict(config["model"])
                config["model"]["model_type"] = "binary"
                config["model"]["num_classes"] = n_cls
                return cls(config, model, device, checkpoint_path=None, mean=mean, std=std)
            raise

        return cls(config, model, device, checkpoint_path=None, mean=mean, std=std)

    def _tensor_from_image_path(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        batch = self._tensor_from_image_path(path).unsqueeze(0).to(self.device)
        return self._forward_batch(batch, str(path))

    def _forward_batch(self, batch: torch.Tensor, path_label: str) -> Dict[str, Any]:
        outputs = self.model(batch)

        if self.model_type == "binary":
            logits = outputs
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            p = probs[0].detach().cpu().numpy()
            normal_idx = next(
                (
                    i
                    for i, n in enumerate(self.class_names)
                    if "正常" in n or "normal" in n.lower() or "Normal" in n
                ),
                None,
            )
            if normal_idx is None and len(p) == 3:
                normal_idx = 1
            p_normal = (
                float(p[normal_idx])
                if normal_idx is not None and normal_idx < len(p)
                else 0.0
            )
            rough_sev = (1.0 - p_normal) * 90.0
            out: Dict[str, Any] = {
                "image_path": path_label,
                "model_type": "binary",
                "class_probabilities": {
                    self.class_names[i]: float(probs[0, i].item()) for i in range(probs.shape[1])
                },
                "predicted_class_index": pred_idx,
                "predicted_class": self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx),
                "severity_estimated_percent": round(float(rough_sev), 1),
                "severity_note": "当前为仅分类权重下的粗略百分比（非 multi_task 分档）；重新训练 multi_task 可得到分档概率。",
            }
            return out

        class_logits, severity_logits = outputs
        class_probs = torch.softmax(class_logits, dim=1)
        sev_probs = torch.softmax(severity_logits, dim=1)

        pred_c = int(torch.argmax(class_probs, dim=1).item())
        pred_s = int(torch.argmax(sev_probs, dim=1).item())

        sev_pct = float((sev_probs.cpu().numpy() @ self.severity_bin_centers).item())

        sev_labels = [
            f"约{int(self.severity_bin_centers[i])}%档"
            for i in range(self.severity_classes)
        ]

        return {
            "image_path": path_label,
            "model_type": "multi_task",
            "class_probabilities": {
                self.class_names[i]: float(class_probs[0, i].item()) for i in range(class_probs.shape[1])
            },
            "predicted_class_index": pred_c,
            "predicted_class": self.class_names[pred_c] if pred_c < len(self.class_names) else str(pred_c),
            "severity_bin_probabilities": {
                sev_labels[i]: float(sev_probs[0, i].item()) for i in range(sev_probs.shape[1])
            },
            "severity_predicted_bin_index": pred_s,
            "severity_estimated_percent": round(sev_pct, 1),
            "severity_interpretation": self._severity_text(sev_probs.cpu().numpy()[0], pred_s),
        }

    def _severity_text(self, sev_prob: np.ndarray, pred_bin: int) -> str:
        tier = ["很轻", "较轻", "中等", "较重", "很重"]
        if pred_bin < len(tier):
            base = tier[pred_bin]
        else:
            base = f"档位{pred_bin}"
        pct = float(sev_prob @ self.severity_bin_centers)
        return f"综合估计严重程度约 {pct:.1f}%（{base}，模型分档置信度 {float(sev_prob[pred_bin]):.1%}）"

    @staticmethod
    def format_report(result: Dict[str, Any]) -> str:
        lines = [
            "=" * 52,
            "胸部 X 线 / 肺炎辅助分析（仅辅助，不能替代医生诊断）",
            "=" * 52,
            f"图像: {result.get('image_path', '')}",
            "",
            "【肺炎类型可能性】",
        ]
        for k, v in result.get("class_probabilities", {}).items():
            lines.append(f"  · {k}: {v:.2%}")
        lines.append(f"  → 模型倾向: {result.get('predicted_class', '')}")
        lines.append("")

        if result.get("model_type") == "binary" and "severity_estimated_percent" in result:
            lines.append("【严重程度（粗略，仅分类模型）】")
            lines.append(f"  → 估计严重度约 {result.get('severity_estimated_percent')}%（基于「非正常」总体可能性）")
            lines.append(f"  · {result.get('severity_note', '')}")
            lines.append("")

        if result.get("model_type") == "multi_task":
            lines.append("【严重程度（分档 + 综合百分比）】")
            for k, v in result.get("severity_bin_probabilities", {}).items():
                lines.append(f"  · {k}: {v:.2%}")
            lines.append(
                f"  → 综合估计严重度: 约 {result.get('severity_estimated_percent', 0)}% "
                f"（加权于各档代表百分比）"
            )
            lines.append(f"  · {result.get('severity_interpretation', '')}")
        lines.append("=" * 52)
        return "\n".join(lines)

    @staticmethod
    def to_json(result: Dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False, indent=2)
