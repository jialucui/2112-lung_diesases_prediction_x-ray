"""
Evaluation metrics for pneumonia detection model
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """Calculate and track training metrics"""
    
    def __init__(self, task='classification'):
        self.task = task
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.pos_probs = []
    
    def add_batch(self, predictions, pos_probs, targets):
        """Add batch predictions"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        self.pos_probs.extend(pos_probs)
    
    def calculate_metrics(self):
        """Calculate all metrics"""
        if not self.targets:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        pos_probs = np.array(self.pos_probs)
        
        metrics = {
            'accuracy': float(accuracy_score(targets, predictions)),
            'precision': float(precision_score(targets, predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(targets, predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(targets, predictions, average='weighted', zero_division=0)),
        }
        
        if len(np.unique(targets)) == 2 and len(pos_probs) == len(targets):
            try:
                metrics['roc_auc'] = float(roc_auc_score(targets, pos_probs))
            except Exception:
                metrics['roc_auc'] = 0.0
        
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics