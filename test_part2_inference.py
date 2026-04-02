"""
Part 2: 使用统一推理接口检测图像（需已有 checkpoints/best_model.pth）
"""

import os
import sys

import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("🧪 PART 2: INFERENCE (detect.py 同款接口)")
print("=" * 60)

from src.inference.predictor import PneumoniaPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

ck = os.path.join(project_root, "checkpoints", "best_model.pth")
if not os.path.isfile(ck):
    print(f"⚠️  未找到 {ck}，请先训练或放入权重后再运行本脚本。")
    sys.exit(0)

predictor = PneumoniaPredictor.from_config_file(
    os.path.join(project_root, "configs", "config.yaml"),
    checkpoint_path=ck,
    device=device,
)

candidates = [
    os.path.join(project_root, "1.jpg"),
    os.path.join(project_root, "yihan.jpg"),
]
paths = [p for p in candidates if os.path.isfile(p)]
if not paths:
    print("未找到示例图片（可放 1.jpg 或 yihan.jpg 到项目根目录）。")
    sys.exit(0)

for p in paths:
    r = predictor.predict(p)
    print(PneumoniaPredictor.format_report(r))
    print()

print("✅ Part 2 done.")
