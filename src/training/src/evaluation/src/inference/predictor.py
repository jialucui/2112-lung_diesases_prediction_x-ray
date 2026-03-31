"""Inference module for single image prediction"""

import torch
import cv2
import numpy as np
from typing import Dict
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PneumoniaPredictor:
    """Predictor class for pneumonia detection"""
    
    PNEUMONIA_CLASSES = {0: 'Normal', 1: 'Pneumonia'}
    SEVERITY_CLASSES = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    
    def __init__(self, model, device: str = 'cuda'):
        """Initialize predictor"""
        self.model = model
        self.device = device
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str) -> Dict:
        """Predict pneumonia and severity"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_pil = Image.fromarray(image_rgb)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            binary_logits, severity_logits = self.model(image_tensor)
        
        binary_probs = torch.softmax(binary_logits, dim=1)
        severity_probs = torch.softmax(severity_logits, dim=1)
        
        binary_pred = binary_logits.argmax(dim=1).item()
        severity_pred = severity_logits.argmax(dim=1).item()
        
        binary_conf = binary_probs[0, binary_pred].item()
        severity_conf = severity_probs[0, severity_pred].item()
        
        result = {
            'has_pneumonia': bool(binary_pred),
            'pneumonia_class': self.PNEUMONIA_CLASSES[binary_pred],
            'pneumonia_confidence': binary_conf,
            'severity': self.SEVERITY_CLASSES[severity_pred] if binary_pred else None,
            'severity_confidence': severity_conf if binary_pred else None,
            'image_path': image_path
        }
        
        logger.info(f"Prediction: {result['pneumonia_class']} ({result['pneumonia_confidence']:.2%})")
        if result['has_pneumonia']:
            logger.info(f"Severity: {result['severity']} ({result['severity_confidence']:.2%})")
        
        return result
