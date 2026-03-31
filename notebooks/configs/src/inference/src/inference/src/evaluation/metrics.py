"""
Model evaluation metrics calculation
Includes: Accuracy, Precision, Recall, F1, AUC-ROC, Sensitivity, Specificity
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self, task='binary'):
        self.task = task
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []
    
    def add_batch(self, predictions, probabilities, ground_truth):
        """Add batch predictions"""
        self.predictions.extend(predictions)
        self.probabilities.extend(probabilities)
        self.ground_truth.extend(ground_truth)
    
    def reset(self):
        """Reset stored predictions"""
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []
    
    def calculate_metrics(self):
        """Calculate all metrics"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        probabilities = np.array(self.probabilities)
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1': f1_score(ground_truth, predictions, zero_division=0),
            'specificity': self._calculate_specificity(ground_truth, predictions),
            'sensitivity': recall_score(ground_truth, predictions, zero_division=0),  # Same as recall
        }
        
        # AUC-ROC
        if len(np.unique(ground_truth)) == 2:
            metrics['auc_roc'] = roc_auc_score(ground_truth, probabilities)
        
        # Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(ground_truth, predictions)
        
        # Classification Report
        metrics['classification_report'] = classification_report(
            ground_truth, predictions, output_dict=True
        )
        
        return metrics
    
    @staticmethod
    def _calculate_specificity(ground_truth, predictions):
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        return specificity
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, ground_truth, probabilities, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(ground_truth, probabilities)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def print_metrics(self, metrics):
        """Pretty print metrics"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
        print("="*50)
