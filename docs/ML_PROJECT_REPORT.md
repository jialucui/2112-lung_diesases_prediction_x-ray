# Lung X-ray ML Project — Technical Report

## 1. 最终完成的任务（分类目标）

| 数据集 | 任务 | 状态 |
|--------|------|------|
| **Chest_xray_data1/chest_xray**（主部署模型） | **NORMAL vs PNEUMONIA**（二分类） | ✅ 已训练并保存 |
| **folder_3** | normal vs pneumonia（二分类，含 train/val/test） | ✅ 可训练 |
| **data_folder_2** / **pneumonia/** | 病毒 vs 细菌（或更多文件夹=多类） | ✅ 代码支持，需对应数据 |
| 三分类 Normal + Virus + Bacteria | 需三个类别文件夹同时存在 | ⚙️ 配置 `num_classes: 3` |

**结论：** 当前**已部署、已评估**的模型是 **NORMAL vs PNEUMONIA**（Chest X-ray）。病毒/细菌分型需使用 `pneumonia/` 下 Virus/Bacteria 子文件夹重新训练；推理端会在三分类模型上额外输出 `subtype` 条件概率。

---

## 2. Age-aware multimodal model

| 项目 | 说明 |
|------|------|
| **架构** | ✅ 已实现：`DenseNetMultiTask` + `tabular_proj`（`model.tabular_dim > 0` 时融合 age/gender） |
| **Chest 已训模型** | ❌ **未使用**：`checkpoints/chest_xray_stopped_epoch7/best_model.pth` 为 **`tabular_dim: 0`（仅图像）** |
| **folder_3** | ⚙️ 支持：`metadata_csv: folder_3/Data_Entry_2017.csv`（列 `Image Index`, `Patient Age`, `Patient Gender`），设 `tabular_dim: 3` 后重新训练 |
| **未在 Chest 上启用原因** | ① Chest 训练时未打开 tabular；② 推理时未提供 age/gender 则向量全 0，与未融合等价 |

**Web/API：** 已支持表单字段 `age`、`gender`；仅当 checkpoint 与 `tabular_dim` 一致且训练时见过表格特征时才有意义。

---

## 3. Confidence score / uncertain case triage

| 项目 | 值 |
|------|-----|
| **Confidence** | `max(class_probabilities)`，字段名 `confidence_score` |
| **Threshold** | **0.8**（`inference.confidence_threshold` in YAML） |
| **规则** | `needs_manual_review = True` 当 `confidence_score < 0.8` |
| **文案** | `inference.triage_message` |

---

## 4. 模型

- **名称：** DenseNet121（`model.name: densenet121`）
- **Pretrained：** **是**（ImageNet，`pretrained: true`）
- **类型：** `multi_task`（分类头 + 严重程度头）

---

## 5. 超参数（Chest X-ray 训练）

| 项 | 值 |
|----|-----|
| 输入尺寸 | **224×224** |
| 输出类别 | **2**（NORMAL, PNEUMONIA） |
| Loss | `0.6 × CrossEntropy(分类) + 0.4 × CrossEntropy(严重程度)` |
| Optimizer | **AdamW** |
| Learning rate | **0.001** |
| Weight decay | **1e-5** |
| Batch size | **32** |
| Epochs（计划/实际） | 20 / **7（手动停止）** |
| Device | **CPU**（本机无 CUDA） |

---

## 6. ML Workflow

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

## 7. 训练结果（Chest_xray，test set，624 张）

| 指标 | 值 |
|------|-----|
| **Accuracy** | **86.06%** |
| **Precision** (macro) | 90.35% |
| **Recall** (macro) | 81.58% |
| **F1** | **89.92%** |
| **ROC-AUC** | **95.69%** |
| **Sensitivity** | 99.49% |
| **Specificity** | 63.68% |

**Confusion matrix（行=真实，列=预测；0=NORMAL，1=PNEUMONIA）：**

|  | Pred NORMAL | Pred PNEUMONIA |
|--|-------------|----------------|
| **True NORMAL** | 149 | 85 |
| **True PNEUMONIA** | 2 | 388 |

验证集（16 张，epoch 5）：Accuracy 87.5%，F1 0.889，AUC 0.969。

完整 YAML：`checkpoints/chest_xray_stopped_epoch7/test_metrics.yaml`  
报告目录：`outputs/chest_xray_report/`（运行 `scripts/generate_ml_report.py` 生成）

---

## 8. Loss / accuracy 曲线

- 来源：`logs/train_chest_xray_20ep.log`（解析脚本在 `src/evaluation/reporting.py`）
- 生成图：`outputs/chest_xray_report/train_loss_curve.png`、`val_loss_curve.png`、`val_f1_curve.png`
- 新训练会自动写入：`logs/training_history.jsonl`

---

## 9. Error analysis

| 现象 | 解释 |
|------|------|
| **NORMAL → PNEUMONIA（85 例）** | 主要错误；正常胸片被标为肺炎，可能因轻度致密影、数据噪声或类别不平衡（肺炎类样本更多） |
| **PNEUMONIA → NORMAL（2 例）** | 较少；模型对肺炎敏感度高（高 sensitivity） |
| **低置信度样本** | `confidence < 0.8` 的样本建议人工复核；常与两类别概率接近（~0.5–0.7）有关 |

详细 JSON：`outputs/chest_xray_report/error_analysis.json`  
逐张预测：`outputs/chest_xray_report/test_predictions.jsonl`

---

## 10. 保存的模型路径

| 用途 | 路径 |
|------|------|
| **推荐（Chest 二分类）** | `checkpoints/chest_xray_stopped_epoch7/best_model.pth` |
| 通用默认（可能被覆盖） | `checkpoints/best_model.pth` |
| 摘要 | `checkpoints/chest_xray_stopped_epoch7/run_summary.yaml` |

---

## 11. 主要代码文件

| 文件 | 用途 |
|------|------|
| `src/training/train.py` | 训练、验证、保存 checkpoint、训练历史 jsonl |
| `src/models/medical_models.py` | DenseNet121 multi-task / binary |
| `src/preprocessing/dicom_xray_loader.py` | 数据加载、增强、metadata/tabular |
| `src/inference/predictor.py` | 推理、confidence、triage |
| `src/inference/cli.py` / `detect.py` | 命令行检测 |
| `src/evaluation/reporting.py` | 完整评估、曲线、错误分析 |
| `scripts/generate_ml_report.py` | 一键生成报告与 demo |
| `web/app.py` + `web/static/index.html` | Web 上传与可视化 |

---

## 12. Web / CLI Demo

**Web：** `python run.py` → http://127.0.0.1:8000  
显示：预测类别、**Confidence %**、threshold 0.8、是否 Flagged for review。

**CLI：**
```bash
python detect.py Chest_xray_data1/chest_xray/test/NORMAL/IM-0031-0001.jpeg \
  --config src/configs/config_chest_xray.yaml \
  --checkpoint checkpoints/chest_xray_stopped_epoch7/best_model.pth
```

**Demo 产物（生成后）：**
- `outputs/chest_xray_report/demo/sample_input.jpg`
- `outputs/chest_xray_report/demo/sample_prediction.json`

（课程报告可截屏 Web 结果页或打开上述 demo 文件。）

---

## 13. 问题与解决

| 问题 | 解决 |
|------|------|
| 数据集路径不统一（folder_3 / Chest_xray / pneumonia） | 支持文件夹类别布局 + `train/val/test` 预设划分 |
| 无 GPU | `--device cpu`；训练较慢 |
| multi_task 缺 severity 标签 | `severity_strategy: auto` 按类别合成 |
| val 集过小（Chest val=16） | 以 test 624 张为主评估；注意 val 波动大 |
| `test_metrics.yaml` 保存失败 | `_metrics_for_yaml` 转换 numpy 类型 |
| 重复训练进程写同一 checkpoint | 单进程 + 按数据集归档目录 |
| 类别名与 config 不一致 | `config_chest_xray.yaml` 单独部署配置 |

---

## 快速命令

```bash
# 生成报告 + 曲线 + 错误分析 + demo
python scripts/generate_ml_report.py

# 启动 Web（默认加载 Chest 模型）
LUNG_XRAY_CHECKPOINT=checkpoints/chest_xray_stopped_epoch7/best_model.pth python run.py
```
