"""
Part 1: Import and Feasibility Test
测试模型导入、初始化和基本功能
"""

import sys
import os
import torch
import yaml

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
    model_multi = DenseNetMultiTask(
        num_classes=3,
        severity_classes=5,
        pretrained=True
    ).to(device)
    params = count_parameters(model_multi)
    print(f"✅ DenseNetMultiTask initialized ({params:,} parameters)")
except Exception as e:
    print(f"❌ Failed to initialize DenseNetMultiTask: {e}")
    sys.exit(1)

try:
    model_binary = BinaryClassifier(
        backbone='densenet121',
        pretrained=True,
        num_classes=3,
    ).to(device)
    params = count_parameters(model_binary)
    print(f"✅ BinaryClassifier initialized ({params:,} parameters)")
except Exception as e:
    print(f"❌ Failed to initialize BinaryClassifier: {e}")
    sys.exit(1)

try:
    model = create_model(
        model_type='multi_task',
        backbone='densenet121',
        pretrained=True,
        device=device,
        num_classes=3,
        severity_classes=5,
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
        class_logits, severity_logits = outputs
        print("✅ Forward pass successful!")
        print(f"   Class logits shape: {class_logits.shape}")
        print(f"   Severity logits shape: {severity_logits.shape}")
    else:
        print(f"❌ Unexpected output type: {type(outputs)}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    sys.exit(1)

# 测试 4: PneumoniaPredictor 初始化（与 configs/config.yaml 一致）
print("\n[Test 4] Initializing PneumoniaPredictor...")
try:
    cfg_path = os.path.join(project_root, "configs", "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    predictor = PneumoniaPredictor(
        config,
        model,
        device,
        checkpoint_path=None,
    )
    print("✅ PneumoniaPredictor initialized successfully")
    print(f"   Device: {predictor.device}")
    print(f"   Class names: {predictor.class_names}")
except Exception as e:
    print(f"❌ Failed to initialize PneumoniaPredictor: {e}")
    sys.exit(1)

# 测试 5: 检查数据目录（可选）
print("\n[Test 5] Checking local data directory...")
data_dir = os.path.join(project_root, "pneumonia_test_folder_1")
if os.path.isdir(data_dir):
    print(f"✅ Data directory found: {data_dir}")
else:
    print(f"⚠️  Optional data directory not found: {data_dir}")

print("\n" + "=" * 60)
print("✅ PART 1 COMPLETED: All imports and basic tests passed!")
print("=" * 60)
print("\n[Info] Run inference: python detect.py your_image.jpg")
print(f"Device: {device}")
print(f"Total parameters: {count_parameters(model):,}")
