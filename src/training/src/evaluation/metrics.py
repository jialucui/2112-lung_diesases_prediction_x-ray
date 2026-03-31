"""Evaluation metrics for pneumonia detection model"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def calculate_metrics(self,
                         binary_preds: List[int],
                         binary_targets: List[int],
                         severity_preds: List[int],
                         severity_targets: List[int]) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {}
        
        # Binary classification metrics
        metrics['binary_accuracy'] = accuracy_score(binary_targets, binary_preds)
        metrics['binary_precision'] = precision_score(binary_targets, binary_preds, zero_division=0)
        metrics['binary_recall'] = recall_score(binary_targets, binary_preds, zero_division=0)
        metrics['binary_f1'] = f1_score(binary_targets, binary_preds, zero_division=0)
        
        # Severity metrics
        metrics['severity_accuracy'] = accuracy_score(severity_targets, severity_preds)
        metrics['severity_f1_macro'] = f1_score(severity_targets, severity_preds, average='macro', zero_division=0)
        metrics['severity_f1_weighted'] = f1_score(severity_targets, severity_preds, average='weighted', zero_division=0)
        
        # Confusion matrices
        metrics['binary_cm'] = confusion_matrix(binary_targets, binary_preds)
        metrics['severity_cm'] = confusion_matrix(severity_targets, severity_preds)
        
        return metrics
