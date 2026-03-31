import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for pneumonia detection"""

    def __init__(self, task='binary'):
        self.task = task
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []

    def add_batch(self, predictions, probabilities, ground_truth):
        """Add batch predictions and ground truth labels"""
        self.predictions.extend(predictions)
        self.probabilities.extend(probabilities)
        self.ground_truth.extend(ground_truth)

    def reset(self):
        """Reset stored predictions"""
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []

    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        probabilities = np.array(self.probabilities)

        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1': f1_score(ground_truth, predictions, zero_division=0),
            'specificity': self._calculate_specificity(ground_truth, predictions),
            'sensitivity': recall_score(ground_truth, predictions, zero_division=0),
        }

        if len(np.unique(ground_truth)) == 2:
            metrics['auc_roc'] = roc_auc_score(ground_truth, probabilities)

        metrics['confusion_matrix'] = confusion_matrix(ground_truth, predictions)
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

    def print_metrics(self, metrics):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
        print("="*50 + "\n")
