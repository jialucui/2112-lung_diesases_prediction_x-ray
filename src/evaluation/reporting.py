"""Evaluation reports: metrics, plots, error analysis, low-confidence triage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import MetricsCalculator
from src.training.train import PneumoniaTrainer, _metrics_for_yaml


def predict_loader_detailed(
    trainer: PneumoniaTrainer,
    loader: DataLoader,
    class_names: List[str],
    confidence_threshold: float = 0.8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run inference on loader; return aggregate metrics + per-image records."""
    trainer.model.eval()
    records: List[Dict[str, Any]] = []
    calc = MetricsCalculator(task="classification")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Detailed eval"):
            images = batch["image"].to(trainer.device)
            labels = batch["label"].cpu().numpy()
            paths = batch.get("image_path", [""] * len(labels))

            outputs = trainer._forward(batch)
            if trainer.model_type == "binary":
                logits = outputs
            else:
                logits, _ = outputs

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            if probs.shape[1] == 2:
                pos_probs = probs[:, 1]
            else:
                pos_probs = probs[np.arange(len(preds)), preds]

            calc.add_batch(preds.tolist(), pos_probs.tolist(), labels.tolist())

            for i in range(len(labels)):
                pred_i = int(preds[i])
                conf = float(probs[i, pred_i])
                true_i = int(labels[i])
                rec = {
                    "image_path": paths[i] if i < len(paths) else "",
                    "true_label": true_i,
                    "true_class": class_names[true_i] if true_i < len(class_names) else str(true_i),
                    "predicted_label": pred_i,
                    "predicted_class": class_names[pred_i] if pred_i < len(class_names) else str(pred_i),
                    "confidence": conf,
                    "correct": pred_i == true_i,
                    "needs_manual_review": conf < confidence_threshold,
                    "class_probabilities": {
                        class_names[j]: float(probs[i, j]) for j in range(probs.shape[1])
                    },
                }
                records.append(rec)

    metrics = calc.calculate_metrics()
    return metrics, records


def error_analysis(records: List[Dict[str, Any]], class_names: List[str]) -> Dict[str, Any]:
    """Summarize misclassifications and low-confidence patterns."""
    wrong = [r for r in records if not r["correct"]]
    uncertain = [r for r in records if r["needs_manual_review"]]
    uncertain_wrong = [r for r in uncertain if not r["correct"]]

    by_true: Dict[str, Dict[str, int]] = {c: {} for c in class_names}
    for r in wrong:
        tc = r["true_class"]
        pc = r["predicted_class"]
        by_true.setdefault(tc, {})
        by_true[tc][pc] = by_true[tc].get(pc, 0) + 1

    conf_wrong = [r["confidence"] for r in wrong]
    conf_right = [r["confidence"] for r in records if r["correct"]]
    conf_uncertain = [r["confidence"] for r in uncertain]

    low_conf_examples = sorted(uncertain, key=lambda x: x["confidence"])[:25]

    return {
        "total_samples": len(records),
        "num_errors": len(wrong),
        "error_rate": len(wrong) / max(len(records), 1),
        "num_uncertain": len(uncertain),
        "uncertain_rate": len(uncertain) / max(len(records), 1),
        "uncertain_and_wrong": len(uncertain_wrong),
        "confusion_pairs_on_errors": by_true,
        "mean_confidence_correct": float(np.mean(conf_right)) if conf_right else None,
        "mean_confidence_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        "mean_confidence_uncertain": float(np.mean(conf_uncertain)) if conf_uncertain else None,
        "interpretation": _error_interpretation(by_true, class_names, len(uncertain), len(wrong)),
        "low_confidence_examples": low_conf_examples,
    }


def _error_interpretation(
    by_true: Dict[str, Dict[str, int]], class_names: List[str], n_uncertain: int, n_wrong: int
) -> str:
    lines = []
    for true_c, preds in by_true.items():
        if not preds:
            continue
        top = max(preds.items(), key=lambda x: x[1])
        lines.append(f"{true_c} most often confused as {top[0]} ({top[1]} cases).")
    if n_uncertain:
        lines.append(
            f"{n_uncertain} samples flagged below confidence threshold; "
            "these often include borderline opacity or label noise."
        )
    if not lines:
        lines.append("No misclassifications on this split.")
    return " ".join(lines)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    title_size = 22
    label_size = 18
    tick_size = 16
    annot_size = 20

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": annot_size, "weight": "bold"},
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title("Confusion Matrix", fontsize=title_size, fontweight="bold", pad=16)
    ax.set_ylabel("True label", fontsize=label_size, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=label_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_size)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: List[int], y_score: List[float], out_path: Path, auc: float) -> None:
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_training_log(log_path: Path) -> Dict[str, List[float]]:
    """Parse tqdm log lines for train loss and val metrics."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []
    val_f1s: List[float] = []

    for m in re.finditer(r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+)", text):
        train_losses.append(float(m.group(2)))

    for block in re.finditer(
        r"Val Loss: ([\d.]+), Val F1: ([\d.]+)(.*?)(?=Epoch|\Z)", text, re.DOTALL
    ):
        val_losses.append(float(block.group(1)))
        val_f1s.append(float(block.group(2)))
        # accuracy not always logged; approximate from F1 if missing
        val_accs.append(float(block.group(2)))

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_f1": val_f1s,
        "val_accuracy_proxy": val_accs,
    }


def merge_training_histories(*histories: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Merge multiple parsed logs by epoch order (dedupe same epoch index)."""
    merged: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy_proxy": [],
    }
    seen_train: set[int] = set()
    seen_val: set[int] = set()
    train_epoch = 0
    val_epoch = 0

    for h in histories:
        for tl in h.get("train_loss", []):
            train_epoch += 1
            if train_epoch not in seen_train:
                merged["train_loss"].append(tl)
                seen_train.add(train_epoch)
        for vl, vf in zip(h.get("val_loss", []), h.get("val_f1", [])):
            val_epoch += 5  # first val at epoch 5 with eval_freq=5
            # val points align to epochs 5,10,15...
            ep = len(merged["val_loss"]) * 5 + 5
            if ep not in seen_val:
                merged["val_loss"].append(vl)
                merged["val_f1"].append(vf)
                merged["val_accuracy_proxy"].append(vf)
                seen_val.add(ep)
    return merged


def plot_training_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = len(history["train_loss"])
    epochs_train = list(range(1, n_train + 1))

    if history["train_loss"]:
        plt.figure(figsize=(9, 5))
        plt.plot(epochs_train, history["train_loss"], marker="o", linewidth=2, label="Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve (Chest X-ray)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "train_loss_curve.png", dpi=150)
        plt.close()

    if history["val_loss"]:
        eval_epochs = [5 * (i + 1) for i in range(len(history["val_loss"]))]
        plt.figure(figsize=(9, 5))
        plt.plot(eval_epochs, history["val_loss"], marker="s", linewidth=2, label="Validation loss", color="C1")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve (Chest X-ray)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "val_loss_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(eval_epochs, history["val_f1"], marker="s", linewidth=2, label="Validation F1", color="C2")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.title("Validation F1 Curve (Chest X-ray)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "val_f1_curve.png", dpi=150)
        plt.close()

    # Combined train + validation loss on one figure
    if history["train_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_train, history["train_loss"], marker="o", linewidth=2, label="Train loss")
        if history["val_loss"]:
            eval_epochs = [5 * (i + 1) for i in range(len(history["val_loss"]))]
            plt.plot(eval_epochs, history["val_loss"], marker="s", linewidth=2, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss (Chest X-ray, NORMAL vs PNEUMONIA)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "train_val_loss_combined.png", dpi=150)
        plt.close()


def generate_full_report(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str = "test",
    confidence_threshold: float = 0.8,
    device: str = "cpu",
    training_log: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write metrics YAML, plots, error analysis, and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer = PneumoniaTrainer(cfg, device=device)
    trainer.load_checkpoint(checkpoint_path)

    dcfg = cfg["data"]
    from src.preprocessing.dicom_xray_loader import create_data_loaders

    repo_root = config_path.resolve().parents[2]
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=dcfg["data_dir"],
        csv_file=dcfg.get("csv_file"),
        batch_size=int(cfg["training"]["batch_size"]),
        image_size=int(dcfg.get("image_size", 224)),
        train_split=dcfg.get("train_split", 0.7),
        val_split=dcfg.get("val_split", 0.15),
        test_split=dcfg.get("test_split", 0.15),
        num_workers=0,
        augment_train=False,
        seed=int(dcfg.get("seed", 42)),
        severity_strategy=dcfg.get("severity_strategy", "auto"),
        synthetic_severity_by_class=dcfg.get("synthetic_severity_by_class"),
        severity_classes=int(cfg["model"].get("severity_classes", 5)),
        metadata_csv=dcfg.get("metadata_csv"),
        tabular_dim=int(cfg["model"].get("tabular_dim", 0)),
        tabular_extra_columns=dcfg.get("tabular_extra_columns"),
        project_root=repo_root,
    )

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(split, test_loader)
    if loader is None:
        raise ValueError(f"No loader for split={split}")

    class_names = trainer.config.get("inference", {}).get("class_display_names")
    if not class_names:
        n = int(cfg["model"]["num_classes"])
        class_names = [f"class_{i}" for i in range(n)]

    metrics, records = predict_loader_detailed(trainer, loader, class_names, confidence_threshold)
    err = error_analysis(records, class_names)

    metrics_yaml = _metrics_for_yaml(metrics)
    with open(output_dir / f"{split}_metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_yaml, f, sort_keys=False)

    with open(output_dir / f"{split}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(err, f, indent=2, ensure_ascii=False)

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    if metrics.get("auc_roc") and trainer.metrics.probabilities:
        plot_roc(
            trainer.metrics.ground_truth,
            trainer.metrics.probabilities,
            output_dir / "roc_curve.png",
            float(metrics["auc_roc"]),
        )

    if training_log and training_log.is_file():
        hist = parse_training_log(training_log)
        with open(output_dir / "training_history_from_log.json", "w") as f:
            json.dump(hist, f, indent=2)
        plot_training_curves(hist, output_dir)

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": split,
        "confidence_threshold": confidence_threshold,
        "metrics": {k: metrics_yaml[k] for k in ("accuracy", "precision", "recall", "f1", "auc_roc", "sensitivity", "specificity") if k in metrics_yaml},
        "error_analysis": {
            "num_errors": err["num_errors"],
            "error_rate": err["error_rate"],
            "num_uncertain": err["num_uncertain"],
            "interpretation": err["interpretation"],
        },
    }
    with open(output_dir / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
