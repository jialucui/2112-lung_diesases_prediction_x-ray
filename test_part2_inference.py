"""
Part 2: Model Inference Testing
使用真实数据测试模型推理
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("🧪 PART 2: MODEL INFERENCE TESTING")
print("=" * 60)

# 导入必要的模块
try:
    from src.models.medical_models import create_model, DenseNetMultiTask
    from src.inference.predictor import PneumoniaPredictor
    from src.preprocessing.dicom_xray_loader import load_image
    print("✅ Successfully imported required modules")
except Exception as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# 配置参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[Config] Using device: {device}")

# ==================== 方案1：使用 Part 1 的模型 ====================
print("\n[Test 1] Loading trained model...")
try:
    # 可以指定模型权重文件
    model_path = os.path.join(project_root, 'models', 'trained_model.pth')
    
    if os.path.exists(model_path):
        # 如果有保存的模型，加载它
        model = torch.load(model_path, map_location=device)
        print(f"✅ Loaded trained model from: {model_path}")
    else:
        # 否则创建一个新的模型（用于演示）
        model = create_model(
            model_type='multi_task',
            backbone='densenet121',
            pretrained=True,
            device=device
        )
        print(f"⚠️  No trained model found. Using pre-trained backbone from ImageNet")
        print(f"   Model path (expected): {model_path}")
    
    model.eval()
    print(f"✅ Model set to evaluation mode")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ==================== 方案2：初始化 Predictor ====================
print("\n[Test 2] Initializing PneumoniaPredictor...")
try:
    predictor = PneumoniaPredictor(model=model, device=device)
    print("✅ PneumoniaPredictor initialized")
except Exception as e:
    print(f"❌ Failed to initialize predictor: {e}")
    sys.exit(1)

# ==================== 方案3：单张图像预测 ====================
def single_image_prediction(image_path):
    """
    单张图像预测
    """
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # 加载并预处理图像
        image = load_image(image_path)  # 使用项目的图像加载函数
        image = image.to(device)
        
        # 推理
        with torch.no_grad():
            binary_logits, severity_logits = model(image.unsqueeze(0))
            
            # 获取预测概率
            binary_probs = torch.softmax(binary_logits, dim=1)
            severity_probs = torch.softmax(severity_logits, dim=1)
            
            # 获取预测结果
            binary_pred = binary_probs.argmax(dim=1).item()
            severity_pred = severity_probs.argmax(dim=1).item()
            
            binary_confidence = binary_probs[0, binary_pred].item()
            severity_confidence = severity_probs[0, severity_pred].item()
        
        result = {
            'image_path': image_path,
            'binary_class': predictor.CLASS_NAMES[binary_pred],
            'binary_confidence': binary_confidence,
            'severity_class': predictor.SEVERITY_NAMES[severity_pred],
            'severity_confidence': severity_confidence,
            'binary_logits': binary_logits.cpu().numpy(),
            'severity_logits': severity_logits.cpu().numpy(),
        }
        
        return result
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

# ==================== 方案4：批量预测 ====================
def batch_prediction(image_paths):
    """
    批量图像预测
    """
    results = []
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not valid_paths:
        print("❌ No valid image paths found")
        return results
    
    try:
        images = []
        for image_path in valid_paths:
            image = load_image(image_path)
            images.append(image)
        
        # 堆叠为批次
        images_batch = torch.stack(images).to(device)
        
        # 批量推理
        with torch.no_grad():
            binary_logits, severity_logits = model(images_batch)
            binary_probs = torch.softmax(binary_logits, dim=1)
            severity_probs = torch.softmax(severity_logits, dim=1)
        
        # 处理结果
        for i, image_path in enumerate(valid_paths):
            binary_pred = binary_probs[i].argmax().item()
            severity_pred = severity_probs[i].argmax().item()
            
            result = {
                'image_path': image_path,
                'binary_class': predictor.CLASS_NAMES[binary_pred],
                'binary_confidence': binary_probs[i, binary_pred].item(),
                'severity_class': predictor.SEVERITY_NAMES[severity_pred],
                'severity_confidence': severity_probs[i, severity_pred].item(),
            }
            results.append(result)
        
        return results
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return results

# ==================== 测试数据 ====================
print("\n[Test 3] Testing on sample images...")

# 定义数据目录（支持多种方式指定）
data_dir = os.environ.get('TEST_DATA_DIR')  # 从环境变量读取
if not data_dir:
    # 尝试相对路径
    data_dir = os.path.join(project_root, 'data', 'test')
    if not os.path.exists(data_dir):
        # 询问用户
        print("⚠️  Test data directory not found")
        print("   Please specify data directory in one of:")
        print("   1. Export TEST_DATA_DIR environment variable")
        print("   2. Place test images in: data/test/")
        print("   3. Modify data_dir in this script")
        data_dir = None

if data_dir and os.path.exists(data_dir):
    # 查找所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.dcm', '.JPG', '.JPEG', '.PNG')
    image_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.lower().endswith(image_extensions)
    ]
    
    if image_files:
        print(f"✅ Found {len(image_files)} test images in: {data_dir}")
        
        # 显示前3张图像的预测结果
        print("\n[Inference Results] (showing first 3):")
        for i, image_path in enumerate(image_files[:3]):
            print(f"\n  Image {i+1}: {os.path.basename(image_path)}")
            result = single_image_prediction(image_path)
            
            if result:
                print(f"    Class: {result['binary_class']} "
                      f"(confidence: {result['binary_confidence']:.4f})")
                print(f"    Severity: {result['severity_class']} "
                      f"(confidence: {result['severity_confidence']:.4f})")
        
        # 批量预测（如果���多张图像）
        if len(image_files) > 1:
            print(f"\n[Batch Test] Running batch prediction on {len(image_files)} images...")
            batch_results = batch_prediction(image_files)
            print(f"✅ Batch prediction completed: {len(batch_results)} results")
    else:
        print(f"⚠️  No image files found in: {data_dir}")
else:
    print("⚠️  Skipping inference test (no test data directory)")

print("\n" + "=" * 60)
print("✅ PART 2 COMPLETED!")
print("=" * 60)
print("\n[Next Steps]")
print("1. Prepare test data in data/test/ directory")
print("2. Or set TEST_DATA_DIR environment variable")
print("3. Run: python test_part2_inference.py")
