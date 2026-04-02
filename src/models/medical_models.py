"""
Medical models for pneumonia detection
Supports: ResNet50, DenseNet121, EfficientNet-B0
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


def create_model(
    model_type: str,
    backbone: str,
    pretrained: bool = True,
    device: str = 'cuda',
    num_classes: Optional[int] = None,
    severity_classes: Optional[int] = None,
):
    """
    Create pneumonia detection model
    
    Args:
        model_type: 'multi_task' or 'binary'
        backbone: 'resnet50', 'densenet121', 'efficientnet-b0'
        pretrained: Whether to use pretrained weights
        device: 'cuda' or 'cpu'
        num_classes: number of output classes for classification head
        severity_classes: number of severity classes for multi-task model
    
    Returns:
        Model instance
    """
    if model_type == 'multi_task':
        model = DenseNetMultiTask(
            num_classes=num_classes or 2,
            severity_classes=severity_classes or 3,
            pretrained=pretrained
        )
    elif model_type == 'binary':
        model = BinaryClassifier(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes or 2
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model.to(device)


class DenseNetMultiTask(nn.Module):
    """Multi-task DenseNet for pneumonia detection and severity prediction"""
    
    def __init__(self, num_classes: int = 2, severity_classes: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Remove classification head
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Severity prediction head
        self.severity_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, severity_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Returns:
            binary_logits: (batch_size, 2)
            severity_logits: (batch_size, severity_classes)
        """
        features = self.backbone(x)
        binary_logits = self.binary_head(features)
        severity_logits = self.severity_head(features)
        return binary_logits, severity_logits


class BinaryClassifier(nn.Module):
    """Binary classifier for pneumonia detection"""
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        
        elif backbone == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
