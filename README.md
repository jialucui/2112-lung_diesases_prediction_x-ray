# Chest X-ray Lung Disease Prediction

Assistive chest X-ray analysis built on **DenseNet121**: **multi-task learning** (disease classification + severity bins), **optional tabular fusion** (age/gender), **confidence-based triage**, **Grad-CAM** explainability, plus a **FastAPI web UI** and **CLI inference**.

> **Disclaimer:** For research and education only. **Not** a substitute for radiologist or clinical diagnosis. Low-confidence outputs should be manually reviewed.

---

## Table of contents

- [Features](#features)
- [Deployed model](#deployed-model)
- [Test metrics: two result sets](#test-metrics-two-result-sets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model and data setup](#model-and-data-setup)
- [Quick start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation scripts](#evaluation-scripts)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [API reference](#api-reference)
- [FAQ](#faq)
- [Further reading](#further-reading)

---

## Features

| Capability | Description |
|------------|-------------|
| **Multi-task** | `multi_task`: class + severity (5 bins, weighted percent) |
| **Binary mode** | `binary`: classification head only (legacy checkpoints) |
| **Folder datasets** | One subfolder per class under `data_dir`; scans `jpg/png/dcm`, etc. |
| **NIH-style CSV** | `metadata_csv` for image name, age, gender, and extras |
| **Tabular fusion** | Fuses age/gender when `model.tabular_dim > 0` |
| **Triage** | `confidence_score < 0.8` → `needs_manual_review` |
| **Grad-CAM** | Heatmap overlays in Web and CLI |
| **Resume training** | `--resume` / `--start-epoch` from checkpoint |
| **Reports** | Metrics, confusion matrix, error analysis, training curves |

---

## Deployed model

Default inference uses **Chest X-ray binary** classification (NORMAL vs PNEUMONIA):

| Item | Value |
|------|-------|
| Config | `src/configs/config_chest_xray.yaml` |
| Weights | `checkpoints/chest_xray_stopped_epoch7/best_model.pth` |
| Normalization | `checkpoints/chest_xray_stopped_epoch7/dataset_norm_stats.json` |
| Input | 224×224 RGB |
| Tabular | **Off** (`tabular_dim: 0`) |

Other setups (`folder_3` three-class, viral/bacterial, etc.) require training with the matching config and checkpoint.

---

## Test metrics: two result sets

The **same checkpoint** (`checkpoints/chest_xray_stopped_epoch7/best_model.pth`, best val F1 at **epoch 5**) was evaluated on the **624-image test split** twice. Numbers differ because **training-time eval (2026-05-16)** used an older preprocessing path; **current scripts** use the aligned inference pipeline.

Full reference: **[docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)**.

### A — Training-time test eval (historical)

Saved when training finished. Source: `checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml`.

| Metric | Value |
|--------|-------|
| Accuracy | 86.06% |
| F1 | 89.92% |
| ROC-AUC | **0.9569** |

**Confusion matrix** (rows = true, cols = predicted; 0=NORMAL, 1=PNEUMONIA):

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| True NORMAL | **149** | **85** |
| True PNEUMONIA | **2** | **388** |

### B — Current pipeline re-eval (figures in `outputs/chest_xray_report/`)

Regenerate: `python scripts/generate_ml_report.py`. Source: `outputs/chest_xray_report/test_metrics.yaml`.

| Metric | Value |
|--------|-------|
| Accuracy | 91.99% |
| F1 | 93.78% |
| ROC-AUC | **0.9773** |

**Confusion matrix:**

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| True NORMAL | **197** | **37** |
| True PNEUMONIA | **13** | **377** |

| Artifact | Result set | Checkpoint |
|----------|------------|------------|
| `confusion_matrix.png` | **B** | `.../chest_xray_stopped_epoch7/best_model.pth` |
| `roc_curve.png` | **B** (AUC 0.9773) | same |
| `test_predictions.jsonl` | **B** | same |

**Do not mix** set A’s matrix (149/85/2/388) with set B’s ROC (0.9773).

---

## Requirements

- **Python** 3.8+
- **PyTorch** 2.x + **torchvision** (install per platform; see below)
- **8GB+ RAM** recommended; GPU optional (`--device cpu` works but is slower)

---

## Installation

```bash
git clone https://github.com/jialucui/2112-lung_diesases_prediction_x-ray.git
cd 2112-lung_diesases_prediction_x-ray

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Install PyTorch:** use the official selector for your CUDA/CPU setup:

https://pytorch.org/get-started/locally/

Example (CPU):

```bash
pip install torch torchvision
```

---

## Model and data setup

### 1. Obtain a checkpoint

`checkpoints/` is gitignored. After cloning you need weights locally:

- Get `checkpoints/chest_xray_stopped_epoch7/best_model.pth` from the maintainer, or  
- Train on `Chest_xray_data1` (see [Training](#training)).

Without weights, Web/CLI fails with `Checkpoint not found`.

### 2. Dataset layouts

#### Option A: Class folders (simplest)

```
your_data/
  train/
    NORMAL/
      *.jpeg
    PNEUMONIA/
      *.jpeg
  val/
    ...
  test/
    ...
```

Or flat multi-class folders (splits applied per config):

```
your_data/
  class_a/
  class_b/
```

Set `data.data_dir` and `data.csv_file: null` in YAML.

#### Option B: NIH-style + CSV (`folder_3`)

```
folder_3/
  train/ ...
  val/ ...
  test/ ...
  Data_Entry_2017.csv
```

Use `src/configs/config.yaml` with `metadata_csv` and `tabular_dim: 3` for age/gender fusion.

#### Supported image formats

`jpg`, `jpeg`, `png`, `dcm`, `bmp`, `tif`, `tiff`, `webp`

Example paths in this repo (if included in your clone):

- `Chest_xray_data1/chest_xray/` — binary NORMAL / PNEUMONIA  
- `folder_3/` — multi-class / NIH metadata  
- `data_folder_2/` — alternate splits  

---

## Quick start

### Web UI (upload + Grad-CAM)

```bash
python run.py
```

Open **http://127.0.0.1:8000** — upload JPG/PNG, view probabilities, severity, confidence, triage flag, and Grad-CAM overlay.

Equivalent:

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### CLI (single image)

```bash
python detect.py Chest_xray_data1/chest_xray/test/NORMAL/IM-0031-0001.jpeg
```

JSON:

```bash
python detect.py path/to/xray.jpg --json
```

Explicit config and checkpoint:

```bash
python detect.py image.jpg \
  --config src/configs/config_chest_xray.yaml \
  --checkpoint checkpoints/chest_xray_stopped_epoch7/best_model.pth
```

Optional patient fields (only if `tabular_dim > 0`):

```bash
python detect.py image.jpg --age 45 --gender M
```

---

## Training

### Chest X-ray binary (matches default deployment)

```bash
python -m src.training.train \
  --config src/configs/config_chest_xray.yaml \
  --device cpu
```

With GPU:

```bash
python -m src.training.train \
  --config src/configs/config_chest_xray.yaml \
  --device cuda
```

### folder_3 three-class + tabular

```bash
python -m src.training.train \
  --config src/configs/config.yaml \
  --device cpu
```

### Common CLI flags

| Flag | Description |
|------|-------------|
| `--config` | YAML config path |
| `--device` | `cpu` or `cuda` |
| `--data-dir` | Override `data.data_dir` |
| `--epochs` | Override `training.num_epochs` |
| `--num-classes` | Override class count (must match folders) |
| `--resume PATH` | Restore model/optimizer/scheduler from `.pth` |
| `--start-epoch N` | Continue from epoch N (0-based) |

Resume from epoch 7 to 20:

```bash
python -m src.training.train \
  --config src/configs/config_chest_xray.yaml \
  --resume checkpoints/chest_xray_stopped_epoch7/last_epoch.pth \
  --start-epoch 7 \
  --epochs 20 \
  --device cpu
```

### Training outputs

| Path | Content |
|------|---------|
| `{checkpoint_dir}/best_model.pth` | Best validation weights |
| `{checkpoint_dir}/last_epoch.pth` | Last epoch (resumable) |
| `{checkpoint_dir}/dataset_norm_stats.json` | Mean/std for inference |
| `{checkpoint_dir}/test_metrics.yaml` | Test metrics |
| `logs/training_history.jsonl` | Per-epoch loss and metrics |

---

## Inference

### Environment variables

| Variable | Purpose |
|----------|---------|
| `LUNG_XRAY_CONFIG` | Override config YAML |
| `LUNG_XRAY_CHECKPOINT` | Override `.pth` path |
| `LUNG_XRAY_NO_DATASET_NORM=1` | Force ImageNet norm (**not recommended**; hurts calibration) |

If unset: `config_chest_xray.yaml` → `config.yaml`; checkpoint from `paths.checkpoint_dir/best_model.pth`.

### Python API

```python
from src.inference.predictor import PneumoniaPredictor

predictor = PneumoniaPredictor.from_config_file(
    "src/configs/config_chest_xray.yaml",
    checkpoint_path="checkpoints/chest_xray_stopped_epoch7/best_model.pth",
    device="cpu",
)
result = predictor.predict("path/to/image.jpeg", confidence_threshold=0.8)
print(predictor.format_report(result))

gc = predictor.grad_cam("path/to/image.jpeg")
# gc["grad_cam_image"] is a base64 PNG data URL
```

### Confidence and triage

- **Confidence** = `max(class_probabilities)`  
- Default threshold **0.8** (`inference.confidence_threshold`)  
- Below threshold: `needs_manual_review: true`  

---

## Evaluation scripts

Run from the **repository root**. Default checkpoint: `checkpoints/chest_xray_stopped_epoch7/best_model.pth`.  
Outputs under `outputs/chest_xray_report/` are **result set B** (see [Test metrics: two result sets](#test-metrics-two-result-sets)).

### Full ML report

```bash
python scripts/generate_ml_report.py \
  --config src/configs/config_chest_xray.yaml \
  --checkpoint checkpoints/chest_xray_stopped_epoch7/best_model.pth
```

Writes to `outputs/chest_xray_report/` by default:

- `test_metrics.yaml` — accuracy, F1, AUC, confusion matrix (**B**)  
- `confusion_matrix.png` — plotted from that YAML  
- `roc_curve.png` — if generated via `plot_roc_curve.py`  
- `test_predictions.jsonl` — per-image probs (source of truth for ROC)  
- `error_analysis.json`, `report_summary.json`  
- `demo/` sample input and prediction JSON (optional)  

### Loss curves only

```bash
python scripts/plot_loss_curves.py \
  --logs logs/train_chest_xray_20ep.log logs/train_chest_xray_7to10.log \
  --output outputs/chest_xray_report
```

### Confusion matrix

```bash
# Plot from existing outputs/chest_xray_report/test_metrics.yaml (result set B)
python scripts/plot_confusion_matrix.py

# Re-run test inference, then plot (refreshes result set B)
python scripts/plot_confusion_matrix.py --reeval
```

Training-time matrix (**149/85/2/388**) is only in `checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml` (result set A).

### ROC curve

```bash
# From test_predictions.jsonl (result set B, AUC matches test_metrics.yaml)
python scripts/plot_roc_curve.py

# Re-run inference first
python scripts/plot_roc_curve.py --reeval
```

### Grad-CAM (single image)

```bash
python scripts/grad_cam_visualize.py \
  --image Chest_xray_data1/chest_xray/test/NORMAL/IM-0031-0001.jpeg \
  --output outputs/grad_cam_overlay.png
```

---

## Configuration

| File | Use case |
|------|----------|
| `src/configs/config_chest_xray.yaml` | **Default deploy:** 2-class Chest X-ray, no tabular |
| `src/configs/config.yaml` | **folder_3:** 3 classes + `metadata_csv` + `tabular_dim: 3` |

Key fields:

```yaml
model:
  model_type: multi_task
  num_classes: 2
  tabular_dim: 0            # 0 = image-only; 3 = [age/100, M, F]

data:
  data_dir: Chest_xray_data1/chest_xray
  norm_stats: checkpoints/.../dataset_norm_stats.json
  metadata_csv: null

inference:
  confidence_threshold: 0.8
```

If severity labels are missing, `severity_strategy: auto` synthesizes bins by class for multi-task training.

---

## Project structure

```
.
├── detect.py                 # CLI shortcut → src.inference.cli
├── run.py                    # Web: uvicorn web.app:app
├── requirements.txt
├── README.md
├── docs/
│   └── ML_PROJECT_REPORT.md
├── scripts/
│   ├── generate_ml_report.py
│   ├── plot_loss_curves.py
│   ├── plot_confusion_matrix.py
│   └── grad_cam_visualize.py
├── src/
│   ├── configs/
│   ├── models/medical_models.py
│   ├── preprocessing/dicom_xray_loader.py
│   ├── training/train.py
│   ├── inference/
│   └── evaluation/
├── web/
│   ├── app.py
│   └── static/index.html
├── Chest_xray_data1/         # sample binary data (if present)
├── folder_3/                 # sample NIH-style data (if present)
├── checkpoints/              # local weights (gitignored)
├── logs/                     # gitignored
└── outputs/                  # gitignored
```

---

## API reference

Base URL: `http://127.0.0.1:8000`. Interactive docs: `/api/v1/docs`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web upload UI |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/info` | Model metadata and endpoints |
| POST | `/api/v1/predict` | `multipart/form-data`: `file` (required), `age`, `gender` (optional) |

Response highlights: `predicted_class`, `class_probabilities`, `confidence`, `needs_manual_review`, `severity_estimated_percent`, `grad_cam_image`, `report`.

---

## FAQ

### `Checkpoint not found`

Place `best_model.pth` under `checkpoints/chest_xray_stopped_epoch7/` or set:

```bash
export LUNG_XRAY_CHECKPOINT=/absolute/path/to/best_model.pth
```

### Web unhealthy / HTTP 503

Check terminal logs: wrong checkpoint path, missing PyTorch, or class count mismatch vs config.

### Confidence looks wrong

Use the same normalization as training (`dataset_norm_stats.json` or `data.norm_stats`). Avoid `LUNG_XRAY_NO_DATASET_NORM=1` unless debugging.

### `tabular_dim` mismatch

The Chest deploy model uses `tabular_dim: 0`. Age/gender in the Web form do not change predictions until you train with `tabular_dim: 3`.

### Training is slow

Expected on CPU. Reduce `batch_size` / `num_workers` or use CUDA.

### Class count vs folders

`model.num_classes` must equal the number of class folders. Override with `--num-classes` if needed.

### Noisy validation metrics

Chest val set is small (16 images); rely on **test** metrics (624 images).

---

## Further reading

- **[docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)** — which checkpoint, which CM, which AUC (sets A vs B)  
- **[docs/METHODOLOGY_QA.md](docs/METHODOLOGY_QA.md)** — dataset size (5856), test 624 source, patient leakage, threshold 0.8, early stopping  
- **[docs/ML_PROJECT_REPORT.md](docs/ML_PROJECT_REPORT.md)** — workflow, hyperparameters, training-time results  
- **Repository:** https://github.com/jialucui/2112-lung_diesases_prediction_x-ray  

---

## License and citation

Follow the license terms of datasets you use (e.g. NIH ChestX-ray14, Kaggle Chest X-ray). Please cite this repository when using the code.

Questions: open a GitHub Issue.
