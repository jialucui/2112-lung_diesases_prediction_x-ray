"""
Part 1: Import and Feasibility Test
测试模型导入、初始化和基本功能
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("🧪 PART 1: IMPORT AND FEASIBILITY TEST")
print("=" * 60)

# 测试 1: 导入模块
print("\n[Test 1] Importing modules...")
try:
    from src.models.medical_models import create_model, DenseNetMultiTask, BinaryClassifier, count_parameters
    print("✅ Successfully imported medical_models")
except Exception as e:
    print(f"❌ Failed to import medical_models: {e}")
    sys.exit(1)

try:
    from src.inference.predictor import PneumoniaPredictor
    print("✅ Successfully imported PneumoniaPredictor")
except Exception as e:
    print(f"❌ Failed to import PneumoniaPredictor: {e}")
    sys.exit(1)

try:
    from src.preprocessing.dicom_xray_loader import load_image, XrayDataset, load_dicom
    print("✅ Successfully imported data loading utilities")
except Exception as e:
    print(f"❌ Failed to import data loading utilities: {e}")
    sys.exit(1)

# 测试 2: 模型初始化
print("\n[Test 2] Initializing models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    # 创建 DenseNetMultiTask 模型
    model_multi = DenseNetMultiTask(
        num_classes=2, 
        severity_classes=3, 
        pretrained=True
    ).to(device)
    params = count_parameters(model_multi)
    print(f"✅ DenseNetMultiTask initialized ({params:,} parameters)")
except Exception as e:
    print(f"❌ Failed to initialize DenseNetMultiTask: {e}")
    sys.exit(1)

try:
    # 创建 BinaryClassifier 模型
    model_binary = BinaryClassifier(
        backbone='densenet121', 
        pretrained=True
    ).to(device)
    params = count_parameters(model_binary)
    print(f"✅ BinaryClassifier initialized ({params:,} parameters)")
except Exception as e:
    print(f"❌ Failed to initialize BinaryClassifier: {e}")
    sys.exit(1)

try:
    # 使用 create_model 函数
    model = create_model(
        model_type='multi_task',
        backbone='densenet121',
        pretrained=True,
        device=device
    )
    print("✅ Model created via create_model() function")
except Exception as e:
    print(f"❌ Failed to create model via function: {e}")
    sys.exit(1)

# 测试 3: 前向传播 (dummy input)
print("\n[Test 3] Forward pass with dummy input...")
try:
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)
    
    if isinstance(outputs, tuple):
        binary_logits, severity_logits = outputs
        print(f"✅ Forward pass successful!")
        print(f"   Binary logits shape: {binary_logits.shape}")
        print(f"   Severity logits shape: {severity_logits.shape}")
    else:
        print(f"❌ Unexpected output type: {type(outputs)}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    sys.exit(1)

# 测试 4: PneumoniaPredictor 初始化
print("\n[Test 4] Initializing PneumoniaPredictor...")
try:
    predictor = PneumoniaPredictor(
        model=model,
        device=device
    )
    print("✅ PneumoniaPredictor initialized successfully")
    print(f"   Device: {predictor.device}")
    print(f"   Class names: {predictor.CLASS_NAMES}")
    print(f"   Severity names: {predictor.SEVERITY_NAMES}")
except Exception as e:
    print(f"❌ Failed to initialize PneumoniaPredictor: {e}")
    sys.exit(1)

# 测试 5: 检查数据文件
print("\n[Test 5] Checking test data directory...")
data_dir = r'C:\Users\tianz\Downloads\2112-lung-test-data1 - Copy'
if os.path.exists(data_dir):
    print(f"✅ Data directory found: {data_dir}")
    
    # 列出数据文件
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm'))]
    print(f"   Found {len(image_files)} image files")
    
    if image_files:
        print(f"   Sample files: {image_files[:3]}")
    else:
        print(f"   ⚠️  No image files found in directory")
else:
    print(f"⚠️  Data directory not found (will be needed for Part 2): {data_dir}")

print("\n" + "=" * 60)
print("✅ PART 1 COMPLETED: All imports and basic tests passed!")
print("=" * 60)

# 保存信息供 Part 2 使用
print("\n[Info] Model ready for inference testing")
print(f"Device: {device}")
print(f"Model type: DenseNetMultiTask")
print(f"Total parameters: {count_parameters(model):,}")
