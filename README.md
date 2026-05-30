# Lung disease prediction (chest X-ray)

Multi-task DenseNet for chest X-ray classification (and optional severity). Includes training, CLI inference, evaluation scripts, and a FastAPI web UI with Grad-CAM.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch separately for your platform if needed: https://pytorch.org/get-started/locally/

### Web UI

```bash
python run.py
# or: uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — upload a chest X-ray for class probabilities, severity estimate, and Grad-CAM overlay.

Environment variables (optional):

| Variable | Purpose |
|----------|---------|
| `LUNG_XRAY_CONFIG` | Override config YAML path |
| `LUNG_XRAY_CHECKPOINT` | Override `.pth` checkpoint |
| `LUNG_XRAY_NO_DATASET_NORM=1` | Force ImageNet normalization |

Defaults use `src/configs/config_chest_xray.yaml` and `checkpoints/chest_xray_stopped_epoch7/best_model.pth` when present.

### CLI

```bash
python detect.py Chest_xray_data1/chest_xray/test/NORMAL/IM-0031-0001.jpeg
python detect.py image.jpg --json
```

### Training (folder_3, 3-class NIH-style layout)

```bash
python -m src.training.train --config src/configs/config.yaml --device cpu
```

Edit `src/configs/config.yaml` for `data.data_dir`, `metadata_csv`, and model settings.

### Training / inference (2-class Chest_xray_data1)

```bash
python -m src.training.train --config src/configs/config_chest_xray.yaml --device cpu
```

## Project layout

| Path | Description |
|------|-------------|
| `src/configs/` | Training & inference YAML |
| `src/training/train.py` | Training loop |
| `src/inference/predictor.py` | Model loading & prediction |
| `web/app.py` | FastAPI + static UI |
| `detect.py` | CLI wrapper |
| `scripts/` | Loss curves, confusion matrix, ML report, Grad-CAM |
| `docs/ML_PROJECT_REPORT.md` | Project report |

`logs/`, `outputs/`, and `checkpoints/` are gitignored (keep checkpoints locally for the web app).

## Data layouts

**Two folders as classes** (virus vs bacteria, or NORMAL vs PNEUMONIA): set `data.csv_file: null` and point `data.data_dir` at the parent folder with one subfolder per class.

**NIH CSV + images**: use `metadata_csv` and `data_dir` as in `src/configs/config.yaml` (`folder_3`).

Supported image types: `jpg`, `jpeg`, `png`, `dcm`, `bmp`, `tif`, `tiff`, `webp`.
