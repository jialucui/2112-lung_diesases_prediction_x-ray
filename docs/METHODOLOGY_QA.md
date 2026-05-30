# Methodology Q&A (dataset, leakage, threshold, early stopping)

Answers below refer to the **deployed Chest X-ray model** (`config_chest_xray.yaml`, `Chest_xray_data1/chest_xray`) unless noted.

---

## Q5 — How many images in the final dataset?

**5,856 images** in total for the Chest X-ray layout used at training time:

| Split | Count | Source |
|-------|------:|--------|
| Train | **5,216** | `Chest_xray_data1/chest_xray/train/` |
| Val | **16** | `Chest_xray_data1/chest_xray/val/` |
| Test | **624** | `Chest_xray_data1/chest_xray/test/` |
| **Total** | **5,856** | |

Logged in `checkpoints/chest_xray_stopped_epoch7/train.log`:

```text
Preset splits: train=5216 val=16 test=624 images
```

**Test class counts** (from `test_metrics.yaml`, support column):

- NORMAL (label 0): **234**
- PNEUMONIA (label 1): **390**
- Sum = **624**

Other datasets in the repo (`folder_3`, `data_folder_2`) are **not** the deployed model; counts differ if you train on them.

---

## Q6 — Where does the 624-image test set come from?

**Not** produced by a random 70/15/15 split in this project for Chest X-ray.

1. Config sets `data.data_dir: Chest_xray_data1/chest_xray`.
2. Loader detects existing **`train/`**, **`val/`**, and **`test/`** subfolders (`_try_collect_preset_train_val_test` in `src/preprocessing/dicom_xray_loader.py`).
3. All images under `test/NORMAL/` and `test/PNEUMONIA/` are collected → **624 files**.

So **624 = the official test folder of the Kaggle-style “Chest X-Ray Images (Pneumonia)” layout** (Mooney, 2018), which matches the common published split:

- Test: 234 normal + 390 pneumonia = 624  
- Train: 5,216 (often cited as 1,341 normal + 3,875 pneumonia)

This project **uses the curator’s folder split as-is**; it does not re-split test from train.

---

## Q7 — Patient-level deduplication? (data leakage)

### What this codebase actually does

| Check | Implemented? |
|-------|----------------|
| **A — Filename duplicate check** | **No** — no dedup by basename across splits |
| **B — Perceptual / file hash dedup** | **No** |
| **C — Patient ID: same patient never in both train and test** | **No in code** — no `Patient ID` column used for Chest config (`metadata_csv: null`) |

Split logic for Chest X-ray:

- **Preset folders** → disjoint by directory (train vs val vs test paths).
- **If preset missing** → random image-level split with `seed=42` (`train_split` / `val_split` / `test_split`) — **would not** group by patient (leakage risk for NIH-style data).

### Honest statement for a medical AI report

**For the deployed model (Chest_xray_data1):**

- Leakage control relies on the **external dataset’s predefined train/test separation**, not on custom patient-level code in this repo.
- The widely used Kaggle Chest X-Ray (Pneumonia) dataset is **described by its authors as using different patients in train vs test** (patient-level split done upstream when the dataset was built).
- **This repository does not verify** that claim (no patient manifest, no hash audit in CI).

**For `folder_3` + NIH `Data_Entry_2017.csv`:**

- CSV contains patient-related fields, but training with **random image split** and `tabular_dim: 3` would require **explicit patient-level grouping** to avoid leakage — **not implemented** today.

### Suggested wording (defensible)

> We used the standard Chest X-Ray (Pneumonia) directory split (train/val/test). Patient-level separation is assumed per the original dataset documentation; our pipeline does not perform additional patient-ID deduplication or cross-split hash checks. Future work should enforce GroupKFold or split-by-patient-ID when using NIH-style metadata.

---

## Q9 — Why is the confidence threshold 0.8?

### What the code does

- `confidence_score = max(class_probabilities)` (argmax class probability).
- `needs_manual_review = True` if `confidence_score < inference.confidence_threshold`.
- Default in YAML: **`0.8`** (`config_chest_xray.yaml`, `config.yaml`).

### How it was chosen (truthful classification)

| Option | Applies? |
|--------|----------|
| **A — Tuned on validation set** | **No** — no script sweeps threshold on val; val has only **16** images (too small to tune reliably). |
| **B — Specific paper citation in repo** | **No** — no reference hard-coded for 0.8. |
| **C — Engineering / clinical convention** | **Yes** — default “high confidence” operating point, common in assistive triage demos. |
| **D — Instructor requirement** | **Unknown** — not documented in repository history. |

Threshold affects **triage flag only**, not training loss, checkpoint selection, or reported accuracy/F1/AUC.

### Suggested wording for report (reasonable packaging)

> We set the manual-review threshold to **0.8** on the maximum predicted class probability as a **conservative operating point**: predictions below 80% confidence are flagged for human review. This follows common practice in computer-aided detection workflows (high-sensitivity screening with human oversight for uncertain cases). We did **not** optimize this threshold on our small validation set (n=16); it can be calibrated on a held-out clinician-labeled set or via cost-sensitive analysis (false negative vs review workload) in deployment.

Optional: cite general CAD literature on human-in-the-loop review rather than claiming val-tuned 0.8.

---

## Q10 — Early stopping (exact settings)

From `src/configs/config_chest_xray.yaml` and `src/training/train.py`:

| Setting | Value |
|---------|--------|
| **`early_stopping_patience`** | **20** |
| **Metric monitored for “best checkpoint”** | Validation **`binary_f1`** (= sklearn **F1**, binary average for 2 classes) |
| **NOT used for best save** | Val loss, val AUC |
| **Validation frequency** | Every **`evaluation.eval_freq: 5`** epochs (epochs 5, 10, 15, 20, …) |
| **Patience counter** | Increments only on eval epochs when val F1 does **not** improve; resets when F1 improves |

Relevant code (`train.py`):

```python
if val_metrics['binary_f1'] > self.best_val_f1:
    self.best_val_f1 = val_metrics['binary_f1']
    self.patience_counter = 0
    self._save_checkpoint(epoch, val_metrics)  # -> best_model.pth
else:
    self.patience_counter += 1
if self.patience_counter >= patience:
    break  # early stop
```

### What actually happened in the Chest run

| Item | Value |
|------|--------|
| Planned epochs | 20 |
| **Completed training epochs** | **7** (manual stop; see `run_summary.yaml`) |
| **Best checkpoint** | **Epoch 5** (`best_checkpoint_epoch: 5`) |
| **Best val F1** | **0.889** (16 val images) |
| Early stop triggered? | **No** — patience 20 was not reached; run stopped before epoch 20 |

So: **checkpoint selection = max val F1**, not val loss; **early stopping was configured but not the reason training ended** (stopped after epoch 7 for other reasons).

---

## Quick reference table (exam / viva)

| Question | Short answer |
|----------|----------------|
| Total images (Chest) | **5,856** (5216 + 16 + 624) |
| Test 624 | **`chest_xray/test/`** preset folder (Kaggle layout) |
| Patient dedup in code | **None**; trust dataset split for Chest; risk if random-split NIH |
| 0.8 threshold | **Default triage constant**, not val-tuned |
| Early stopping | **patience=20**, monitor **val F1**, eval every **5** epochs; best model **epoch 5** |
