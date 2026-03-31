"""
Inference module for pneumonia prediction from chest X-rays
"""

import torch
import numpy as np
from pathlib import Path
from ..models.medical_models import DenseNetMultiTask


class PneumoniaPredictor:
    """Predictor class for pneumonia detection"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        model = DenseNetMultiTask(
            num_classes=2,
            severity_classes=4,
            pretrained=False
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    @torch.no_grad()
    def predict(self, images):
        """
        Predict pneumonia presence and severity
        
        Args:
            images: Tensor of shape (B, C, H, W)
        
        Returns:
            dict: Predictions with probabilities
        """
        images = images.to(self.device)
        
        class_logits, severity_logits = self.model(images)
        
        class_probs = torch.softmax(class_logits, dim=1)
        severity_probs = torch.softmax(severity_logits, dim=1)
        
        class_pred = torch.argmax(class_logits, dim=1)
        severity_pred = torch.argmax(severity_logits, dim=1)
        
        return {
            'pneumonia_prediction': class_pred.cpu().numpy(),
            'pneumonia_probability': class_probs[:, 1].cpu().numpy(),  # Prob of positive class
            'severity_prediction': severity_pred.cpu().numpy(),
            'severity_probabilities': severity_probs.cpu().numpy(),
        }
    
    def predict_batch(self, batch_images, batch_size=32):
        """Process batch of images"""
        predictions = {
            'pneumonia_prediction': [],
            'pneumonia_probability': [],
            'severity_prediction': [],
            'severity_probabilities': [],
        }
        
        for i in range(0, len(batch_images), batch_size):
            batch = batch_images[i:i+batch_size]
            batch_pred = self.predict(batch)
            
            for key in predictions:
                predictions[key].extend(batch_pred[key])
        
        # Convert to arrays
        for key in predictions:
            predictions[key] = np.array(predictions[key])
        
        return predictions
    
    def get_prediction_summary(self, prediction_dict):
        """Generate human-readable summary"""
        class_names = {0: 'Normal', 1: 'Pneumonia'}
        severity_names = {0: 'Mild', 1: 'Moderate', 2: 'Severe', 3: 'Critical'}
        
        summaries = []
        for i in range(len(prediction_dict['pneumonia_prediction'])):
            summary = {
                'diagnosis': class_names[prediction_dict['pneumonia_prediction'][i]],
                'confidence': f"{prediction_dict['pneumonia_probability'][i]:.2%}",
                'severity': severity_names[prediction_dict['severity_prediction'][i]],
                'severity_confidence': f"{prediction_dict['severity_probabilities'][i].max():.2%}"
            }
            summaries.append(summary)
        
        return summaries
