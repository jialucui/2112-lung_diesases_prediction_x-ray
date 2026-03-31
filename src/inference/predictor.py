"""
Inference module for pneumonia prediction from chest X-rays
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class PneumoniaPredictor:
    """Predictor class for pneumonia detection"""
    
    CLASS_NAMES = {0: 'Normal', 1: 'Pneumonia'}
    SEVERITY_NAMES = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    
    def __init__(self, model, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize predictor
        
        Args:
            model: Loaded model instance
            model_path: Path to model checkpoint (for loading state dict)
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()
        
        # Load checkpoint if provided
        if model_path:
            self._load_checkpoint(model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_checkpoint(self, model_path: str):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess image"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        # Read image
        if str(image_path).lower().endswith('.dcm'):
            import pydicom
            dicom_data = pydicom.dcmread(image_path)
            image = dicom_data.pixel_array
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Apply transforms
        image_tensor = self.transform(image_pil)
        return image_tensor
    
    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict pneumonia and severity for single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict with predictions and probabilities
        """
        # Load and preprocess image
        image_tensor = self._load_image(image_path).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor)
        
        # Handle both single and multi-task outputs
        if isinstance(outputs, tuple):
            binary_logits, severity_logits = outputs
        else:
            binary_logits = outputs
            severity_logits = None
        
        # Get predictions
        binary_probs = torch.softmax(binary_logits, dim=1)
        binary_pred = torch.argmax(binary_logits, dim=1).item()
        binary_conf = binary_probs[0, binary_pred].item()
        
        result = {
            'has_pneumonia': bool(binary_pred),
            'pneumonia_class': self.CLASS_NAMES[binary_pred],
            'pneumonia_confidence': binary_conf,
            'image_path': str(image_path)
        }
        
        # Add severity if available
        if severity_logits is not None:
            severity_probs = torch.softmax(severity_logits, dim=1)
            severity_pred = torch.argmax(severity_logits, dim=1).item()
            severity_conf = severity_probs[0, severity_pred].item()
            
            result['severity'] = self.SEVERITY_NAMES.get(severity_pred, 'Unknown')
            result['severity_confidence'] = severity_conf
        
        logger.info(f"Prediction: {result['pneumonia_class']} ({result['pneumonia_confidence']:.2%})")
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> Dict:
        """
        Predict on batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            dict with batch predictions
        """
        all_predictions = []
        all_pneumonia_probs = []
        all_severity_preds = []
        all_severity_probs = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch
            batch_images = []
            for path in batch_paths:
                try:
                    img_tensor = self._load_image(path)
                    batch_images.append(img_tensor)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Forward pass
            outputs = self.model(batch_tensor)
            
            if isinstance(outputs, tuple):
                binary_logits, severity_logits = outputs
            else:
                binary_logits = outputs
                severity_logits = None
            
            # Get predictions
            binary_probs = torch.softmax(binary_logits, dim=1)
            binary_preds = torch.argmax(binary_logits, dim=1).cpu().numpy()
            
            all_predictions.extend(binary_preds)
            all_pneumonia_probs.extend(binary_probs[:, 1].cpu().numpy())
            
            if severity_logits is not None:
                severity_preds = torch.argmax(severity_logits, dim=1).cpu().numpy()
                severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()
                all_severity_preds.extend(severity_preds)
                all_severity_probs.extend(severity_probs)
        
        return {
            'pneumonia_predictions': np.array(all_predictions),
            'pneumonia_probabilities': np.array(all_pneumonia_probs),
            'severity_predictions': np.array(all_severity_preds) if all_severity_preds else None,
            'severity_probabilities': np.array(all_severity_probs) if all_severity_probs else None,
        }
    
    def get_prediction_summary(self, predictions: Dict) -> str:
        """Generate human-readable prediction summary"""
        total = len(predictions.get('pneumonia_predictions', []))
        pneumonia_cases = np.sum(predictions['pneumonia_predictions'])
        normal_cases = total - pneumonia_cases
        
        summary = f"""
{'='*50}
PREDICTION SUMMARY
{'='*50}
Total Predictions: {total}
Normal Cases: {normal_cases} ({100*normal_cases/total:.1f}%)
Pneumonia Cases: {pneumonia_cases} ({100*pneumonia_cases/total:.1f}%)
Average Confidence: {predictions['pneumonia_probabilities'].mean():.2%}
{'='*50}
        """
        return summary
