# Lung X-ray ML Project — Technical Report

> **Metrics & confusion matrices:** This report’s **Section 7** numbers are **training-time (result set A)**.  
> Plots under `outputs/chest_xray_report/` and ROC **AUC 0.9773** are **current-pipeline (result set B)**.  
> See **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)** for a side-by-side reference (same checkpoint, different eval runs).

## 1. Classification goals completed

| Dataset | Task | Status |
|---------|------|--------|
| **Chest_xray_data1/chest_xray** (deployed model) | **NORMAL vs PNEUMONIA** (binary) | Trained and saved |
| **folder_3** | normal vs pneumonia (with train/val/test) | Trainable |
| **data_folder_2** / **pneumonia/** | viral vs bacterial (or more folders = multi-class) | Supported; needs matching data |
| 3-class Normal + Virus + Bacteria | Requires three class folders | Set `num_classes: 3` in config |

**Summary:** The **deployed and evaluated** model is **NORMAL vs PNEUMONIA** (Chest X-ray). Viral/bacterial subtyping requires retraining on `pneumonia/` Virus/Bacteria folders; inference on a 3-class model also outputs conditional `subtype` probabilities.

---

## 2. Age-aware multimodal model

| Item | Details |
|------|---------|
| **Architecture** | Implemented: `DenseNetMultiTask` + `tabular_proj` when `model.tabular_dim > 0` (age/gender fusion) |
| **Chest checkpoint** | **Not used:** `checkpoints/chest_xray_stopped_epoch7/best_model.pth` has **`tabular_dim: 0` (image-only)** |
| **folder_3** | Supported: `metadata_csv: folder_3/Data_Entry_2017.csv` (`Image Index`, `Patient Age`, `Patient Gender`); set `tabular_dim: 3` and retrain |
| **Why not on Chest** | Tabular was off during Chest training; missing age/gender at inference yields zeros (same as no fusion) |

**Web/API:** Form fields `age` and `gender` are accepted; they only affect predictions when the checkpoint was trained with matching `tabular_dim` and tabular features.

---

## 3. Confidence score / uncertain case triage

| Item | Value |
|------|-------|
| **Confidence** | `max(class_probabilities)` → `confidence_score` |
| **Threshold** | **0.8** (`inference.confidence_threshold` in YAML) |
| **Rule** | `needs_manual_review = True` when `confidence_score < 0.8` |
| **Message** | `inference.triage_message` |

---

## 4. Model

- **Backbone:** DenseNet121 (`model.name: densenet121`)
- **Pretrained:** Yes (ImageNet, `pretrained: true`)
- **Type:** `multi_task` (classification head + severity head)

---

## 5. Hyperparameters (Chest X-ray training)

| Item | Value |
|------|-------|
| Input size | **224×224** |
| Classes | **2** (NORMAL, PNEUMONIA) |
| Loss | `0.6 × CrossEntropy(class) + 0.4 × CrossEntropy(severity)` |
| Optimizer | **AdamW** |
| Learning rate | **0.001** |
| Weight decay | **1e-5** |
| Batch size | **32** |
| Epochs (planned / actual) | 20 / **7 (manual stop)** |
| Device | **CPU** (no CUDA on training machine) |

---

## 6. ML workflow

```
Chest X-ray JPG
    → Resize 224, augment (train only)
    → Dataset mean/std normalization
    → DenseNet121 backbone
    → [optional tabular: age/100, M, F]
    → Class head + Severity head
    → Train (AdamW, multi-task loss)
    → Val every 5 epochs → save best_model.pth
    → Test evaluation + error analysis (scripts/generate_ml_report.py)
    → Inference: web/app.py or python detect.py
```

---

## 7. Results (Chest_xray, test set, 624 images)

**Checkpoint:** `checkpoints/chest_xray_stopped_epoch7/best_model.pth` (best validation F1 at epoch 5).

### 7a. Training-time test eval (result set A — historical)

Recorded when training ended (~2026-05-16) via `src/training/train.py` → `test_metrics.yaml` next to the checkpoint.

| Metric | Value |
|--------|-------|
| **Accuracy** | **86.06%** |
| **Precision** (macro) | 90.35% |
| **Recall** (macro) | 81.58% |
| **F1** | **89.92%** |
| **ROC-AUC** | **0.9569** |
| **Sensitivity** | 99.49% |
| **Specificity** | 63.68% |

**Confusion matrix** (rows = true, columns = predicted; 0 = NORMAL, 1 = PNEUMONIA):

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **True NORMAL** (234) | **149** (TN) | **85** (FP) |
| **True PNEUMONIA** (390) | **2** (FN) | **388** (TP) |

File: `checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml`

### 7b. Current pipeline re-eval (result set B — `outputs/chest_xray_report/`)

Same weights, re-run with `scripts/generate_ml_report.py` and today’s preprocessing (`load_xray_rgb`, `dataset_norm_stats.json`).

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.99%** |
| **F1** | **93.78%** |
| **ROC-AUC** | **0.9773** |
| **Sensitivity** | 96.67% |
| **Specificity** | 84.19% |

**Confusion matrix:**

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **True NORMAL** | **197** | **37** |
| **True PNEUMONIA** | **13** | **377** |

Files: `outputs/chest_xray_report/test_metrics.yaml`, `confusion_matrix.png`, `roc_curve.png`, `test_predictions.jsonl`

**Do not mix:** ROC **0.9773** belongs to **7b**, not the **149/85/2/388** matrix in **7a**.

### Validation (not test)

Validation set (**16 images**, epoch 5): Accuracy 87.5%, F1 0.889, AUC 0.969 — `run_summary.yaml` → `val_metrics_at_best`.

---

## 8. Loss / accuracy curves

- Source: `logs/train_chest_xray_20ep.log` (parsed by `src/evaluation/reporting.py`)
- Plots: `outputs/chest_xray_report/train_loss_curve.png`, `val_loss_curve.png`, `val_f1_curve.png`
- New runs also write: `logs/training_history.jsonl`

---

## 9. Error analysis

### Training-time (result set A)

| Pattern | Count | Notes |
|---------|-------|-------|
| **NORMAL → PNEUMONIA** | **85** | Main error in CM 149/85/2/388 |
| **PNEUMONIA → NORMAL** | **2** | Rare; high sensitivity |

### Current re-eval (result set B)

From `outputs/chest_xray_report/error_analysis.json` (same run as CM 197/37/13/377):

| Pattern | Count | Notes |
|---------|-------|-------|
| **NORMAL → PNEUMONIA** | **37** | Most common mistake |
| **PNEUMONIA → NORMAL** | **13** | |
| **Low-confidence** (`confidence < 0.8`) | **32** | Manual review recommended |

Per-image predictions: `outputs/chest_xray_report/test_predictions.jsonl`

---

## 10. Saved model paths

| Purpose | Path |
|---------|------|
| **Recommended (Chest binary)** | `checkpoints/chest_xray_stopped_epoch7/best_model.pth` |
| Generic default (may be overwritten) | `checkpoints/best_model.pth` |
| Run summary | `checkpoints/chest_xray_stopped_epoch7/run_summary.yaml` |

---

## 11. Key source files

| File | Role |
|------|------|
| `src/training/train.py` | Training, validation, checkpoints, `training_history.jsonl` |
| `src/models/medical_models.py` | DenseNet121 multi-task / binary |
| `src/preprocessing/dicom_xray_loader.py` | Loading, augmentation, metadata/tabular |
| `src/inference/predictor.py` | Inference, confidence, triage |
| `src/inference/cli.py` / `detect.py` | CLI |
| `src/evaluation/reporting.py` | Full eval, curves, error analysis |
| `scripts/generate_ml_report.py` | One-shot report + demo |
| `web/app.py` + `web/static/index.html` | Web upload and visualization |

---

## 12. Web / CLI demo

**Web:** `python run.py` → http://127.0.0.1:8000  
Shows predicted class, **confidence %**, threshold 0.8, and flagged-for-review status.

**CLI:**
```bash
python detect.py Chest_xray_data1/chest_xray/test/NORMAL/IM-0031-0001.jpeg \
  --config src/configs/config_chest_xray.yaml \
  --checkpoint checkpoints/chest_xray_stopped_epoch7/best_model.pth
```

**Demo artifacts (after report generation):**
- `outputs/chest_xray_report/demo/sample_input.jpg`
- `outputs/chest_xray_report/demo/sample_prediction.json`

---

## 13. Issues and fixes

| Issue | Resolution |
|-------|------------|
| Mixed dataset paths (folder_3 / Chest_xray / pneumonia) | Folder-based class layout + optional `train/val/test` splits |
| No GPU | `--device cpu`; slower training |
| Missing severity labels for multi_task | `severity_strategy: auto` synthesizes bins by class |
| Small val set (Chest val = 16) | Use test set (624 images) for primary metrics |
| `test_metrics.yaml` save failures | `_metrics_for_yaml` converts numpy types |
| Concurrent training overwriting checkpoints | Single process; per-dataset checkpoint directories |
| Class names vs config mismatch | Separate `config_chest_xray.yaml` for deployment |

---

## Quick commands

```bash
# Full report + curves + error analysis + demo
python scripts/generate_ml_report.py

# Start Web (loads Chest model by default)
LUNG_XRAY_CHECKPOINT=checkpoints/chest_xray_stopped_epoch7/best_model.pth python run.py
```
