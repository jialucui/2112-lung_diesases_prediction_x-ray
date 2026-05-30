# Evaluation results reference (Chest X-ray, test split)

This document explains **two different test-set evaluations** that appear in the repo. They use the **same checkpoint weights** but were produced at **different times** with **different evaluation code paths**, so metrics and confusion matrices **do not match**.

---

## Shared setup

| Item | Value |
|------|-------|
| **Checkpoint** | `checkpoints/chest_xray_stopped_epoch7/best_model.pth` |
| **Best epoch (val F1)** | 5 — see `checkpoints/chest_xray_stopped_epoch7/run_summary.yaml` |
| **Config** | `src/configs/config_chest_xray.yaml` |
| **Classes** | 0 = NORMAL, 1 = PNEUMONIA |
| **Test split** | `Chest_xray_data1/chest_xray/test/` — **624 images** (234 NORMAL + 390 PNEUMONIA) |
| **Training stopped** | After epoch 7 (planned 20); test eval run at end of training (~2026-05-16) |

**Confusion matrix layout** (sklearn / this repo): **rows = true label**, **columns = predicted label**.

```
              Pred NORMAL    Pred PNEUMONIA
True NORMAL      TN              FP
True PNEUMONIA   FN              TP
```

**ROC-AUC:** probability score = **P(PNEUMONIA)** = `class_probabilities["PNEUMONIA"]` (label 1).  
**Hard labels** (accuracy, confusion matrix): **argmax** over class probabilities.

---

## Result set A — training-time test eval (historical)

**When:** End of training (~2026-05-16).  
**How:** `python -m src.training.train` → final `trainer.evaluate(test_loader)` in `src/training/train.py`.  
**Saved to:** `checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml` (and summarized in `run_summary.yaml`).

| Metric | Value |
|--------|-------|
| Accuracy | **86.06%** |
| F1 (binary) | **89.92%** |
| ROC-AUC | **0.9569** |
| Sensitivity | 99.49% |
| Specificity | 63.68% |

**Confusion matrix:**

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **True NORMAL** (234) | **149** (TN) | **85** (FP) |
| **True PNEUMONIA** (390) | **2** (FN) | **388** (TP) |

```yaml
# checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml
confusion_matrix:
  - [149, 85]
  - [2, 388]
auc_roc: 0.9569417050186281
```

**Use this set when:** you need the number that was recorded **at training time** (e.g. coursework tied to the original training run).

**Caveat:** The inference/preprocessing stack in the repo was **improved after** this run (OpenCV load path, `dataset_norm_stats.json`, `resolve_norm_stats`). Re-running today’s scripts on the same checkpoint does **not** reproduce these numbers.

---

## Result set B — current pipeline re-evaluation (recommended for repo artifacts)

**When:** Re-run with current code (~2026-05-24 in this workspace).  
**How:** `python scripts/generate_ml_report.py` (or `plot_confusion_matrix.py --reeval`, `plot_roc_curve.py --reeval`).  
**Implementation:** `src/evaluation/reporting.py` → `predict_loader_detailed` + same metric logic as training eval, but with **current** dataloader and normalization.

| Metric | Value |
|--------|-------|
| Accuracy | **91.99%** |
| F1 (binary) | **93.78%** |
| ROC-AUC | **0.9773** |
| Sensitivity | 96.67% |
| Specificity | 84.19% |

**Confusion matrix:**

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **True NORMAL** (234) | **197** (TN) | **37** (FP) |
| **True PNEUMONIA** (390) | **13** (FN) | **377** (TP) |

**Files (all consistent with each other):**

| File | Role |
|------|------|
| `outputs/chest_xray_report/test_metrics.yaml` | Metrics + CM |
| `outputs/chest_xray_report/test_predictions.jsonl` | Per-image labels and probabilities |
| `outputs/chest_xray_report/confusion_matrix.png` | Plot from `test_metrics.yaml` (default `plot_confusion_matrix.py`) |
| `outputs/chest_xray_report/roc_curve.png` | ROC from `test_predictions.jsonl` (`plot_roc_curve.py`) |
| `outputs/chest_xray_report/report_summary.json` | Checkpoint path + headline metrics |
| `outputs/chest_xray_report/error_analysis.json` | Errors / low-confidence (e.g. 37 NORMAL→PNEUMONIA) |

**Use this set when:** you use **Web / CLI / report scripts** in the current repo, or when citing figures under `outputs/chest_xray_report/`.

**Checkpoint field in `report_summary.json`:**

```json
"checkpoint": ".../checkpoints/chest_xray_stopped_epoch7/best_model.pth"
```

---

## Quick mapping (what goes with what)

| You see… | Result set | Confusion matrix | AUC |
|----------|------------|------------------|-----|
| README / report “149, 85, 2, 388” | **A** (training-time) | 149 / 85 / 2 / 388 | **0.9569** |
| `outputs/.../confusion_matrix.png` (default) | **B** (re-eval) | 197 / 37 / 13 / 377 | — |
| `outputs/.../roc_curve.png` | **B** | Same preds as JSONL | **0.9773** |
| AUC **0.9773** in conversation | **B** | 197 / 37 / 13 / 377 | **0.9773** |

**Do not mix:** e.g. ROC **0.9773** does **not** correspond to the **149/85/2/388** matrix.

---

## How to regenerate result set B

From the repo root (requires local test images under `Chest_xray_data1/chest_xray/test/`):

```bash
# Full report: metrics, CM, ROC, error analysis, demo
python scripts/generate_ml_report.py \
  --config src/configs/config_chest_xray.yaml \
  --checkpoint checkpoints/chest_xray_stopped_epoch7/best_model.pth

# Or individual plots
python scripts/plot_confusion_matrix.py --reeval
python scripts/plot_roc_curve.py --reeval
```

To plot CM/ROC **without** re-inference (only if `test_metrics.yaml` / `test_predictions.jsonl` already exist):

```bash
python scripts/plot_confusion_matrix.py
python scripts/plot_roc_curve.py
```

---

## Why A and B differ (same `best_model.pth`)

1. **Preprocessing alignment** — Current inference uses `load_xray_rgb` + normalization from `dataset_norm_stats.json` (see `resolve_norm_stats` in `src/preprocessing/dicom_xray_loader.py`). Training-time test eval used the code as it existed on **2026-05-16**.
2. **Same argmax rule, different inputs** — Different resized/normalized tensors → different softmax → different CM and AUC.
3. **Not a different model file** — Both paths load `checkpoints/chest_xray_stopped_epoch7/best_model.pth` unless you override `LUNG_XRAY_CHECKPOINT`.

---

## Validation set (separate from test)

At **best checkpoint (epoch 5)**, validation (**16 images**): Accuracy 87.5%, F1 0.889, AUC 0.969 — from `run_summary.yaml` → `val_metrics_at_best`.  
Val is too small for stable reporting; **test (624)** is the primary holdout.
