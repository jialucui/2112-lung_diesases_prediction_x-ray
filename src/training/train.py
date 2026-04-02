"""
Complete training pipeline for pneumonia detection model

Features:
- Multi-task learning
- Mixed precision training
- Early stopping
- Checkpoint saving
- TensorBoard logging
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import yaml
import argparse
import logging
from typing import Dict, Tuple
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.medical_models import create_model, DenseNetMultiTask
from src.preprocessing.dicom_xray_loader import create_data_loaders
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PneumoniaTrainer:
    """Trainer class for pneumonia detection model"""
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        """Initialize trainer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.model_type = self.config['model'].get('model_type', 'multi_task')
        
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self.model = create_model(
            model_type=self.config['model']['model_type'],
            backbone=self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            device=device,
            num_classes=self.config['model'].get('num_classes'),
            severity_classes=self.config['model'].get('severity_classes'),
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        # Gradient scaler (enabled only when mixed precision is requested and CUDA is available)
        mp_requested = bool(self.config['training'].get('mixed_precision', True))
        self.use_mixed_precision = mp_requested and torch.cuda.is_available() and device.startswith('cuda')
        self.scaler = GradScaler(enabled=self.use_mixed_precision)
        
        # Metrics calculator (binary or multiclass)
        self.metrics = MetricsCalculator(task='classification')
        
        logger.info(f"Trainer initialized on {device}")
    
    def _compute_loss(self, outputs, batch):
        """Compute loss for binary or multi-task model."""
        labels = batch['label'].to(self.device)

        if self.model_type == 'binary':
            logits = outputs
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, {'loss': loss.detach()}

        # multi_task
        binary_logits, severity_logits = outputs
        binary_labels = labels
        if 'severity' not in batch:
            raise ValueError("Multi-task training requires 'severity' labels in the dataset/batch.")
        severity_labels = batch['severity'].to(self.device)

        binary_loss = nn.CrossEntropyLoss()(binary_logits, binary_labels)
        severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)

        lw = self.config['training']['loss_weights']
        binary_weight = lw.get('classification', lw.get('binary_classification', 0.6))
        severity_weight = lw['severity_prediction']

        total_loss = binary_weight * binary_loss + severity_weight * severity_loss
        return total_loss, {
            'loss': total_loss.detach(),
            'binary_loss': binary_loss.detach(),
            'severity_loss': severity_loss.detach(),
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)

            with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                outputs = self.model(images)
                loss, _ = self._compute_loss(outputs, batch)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_max_norm']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss / (progress_bar.n + 1):.4f}"})
        
        return {'loss': total_loss / len(train_loader)}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_pos_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)

                with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                    outputs = self.model(images)
                    loss, _ = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()

                labels = batch['label'].cpu().numpy()

                if self.model_type == 'binary':
                    logits = outputs
                else:
                    logits, _ = outputs  # use binary head for reporting

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs[np.arange(len(preds)), preds]

                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                all_pos_probs.extend(pos_probs.tolist())

        self.metrics.reset()
        self.metrics.add_batch(all_preds, all_pos_probs, all_targets)
        metrics = self.metrics.calculate_metrics()
        metrics['loss'] = total_loss / len(val_loader)

        # keep compatibility with old logging key name used elsewhere
        metrics['binary_f1'] = metrics.get('f1', 0.0)
        return metrics

    def evaluate(self, loader: DataLoader, name: str = "test") -> Dict:
        """Evaluate a loader and return metrics dict."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_pos_probs = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({name})"):
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()

                with autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu', enabled=self.use_mixed_precision):
                    outputs = self.model(images)

                if self.model_type == 'binary':
                    logits = outputs
                else:
                    logits, _ = outputs

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs[np.arange(len(preds)), preds]

                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                all_pos_probs.extend(pos_probs.tolist())

        self.metrics.reset()
        self.metrics.add_batch(all_preds, all_pos_probs, all_targets)
        metrics = self.metrics.calculate_metrics()
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            if (epoch + 1) % self.config['evaluation']['eval_freq'] == 0:
                val_metrics = self.validate(val_loader, epoch)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['binary_f1']:.4f}")
                
                if val_metrics['binary_f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['binary_f1']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(f"✅ Best model saved! F1: {self.best_val_f1:.4f}")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.scheduler.step()
        
        logger.info("✅ Training completed!")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1
        }
        
        checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument(
    '--device',
    type=str,
    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    trainer = PneumoniaTrainer(args.config, device=args.device)
    
    dcfg = trainer.config['data']
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=dcfg['data_dir'],
        csv_file=dcfg.get('csv_file'),
        batch_size=trainer.config['training']['batch_size'],
        image_size=dcfg.get('image_size', 224),
        train_split=dcfg.get('train_split', 0.7),
        val_split=dcfg.get('val_split', 0.15),
        test_split=dcfg.get('test_split', 0.15),
        num_workers=dcfg['num_workers'],
        augment_train=dcfg['augment_train'],
        seed=dcfg['seed'],
        severity_strategy=dcfg.get('severity_strategy', 'none'),
        synthetic_severity_by_class=dcfg.get('synthetic_severity_by_class'),
    )
    
    trainer.train(train_loader, val_loader)

    # Optional: evaluate on test split if present
    if test_loader is not None:
        # Load best checkpoint for evaluation if available
        best_path = trainer.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(ckpt['model_state_dict'])
        test_metrics = trainer.evaluate(test_loader, name="test")
        logger.info(f"Test - Acc: {test_metrics.get('accuracy', 0.0):.4f}, F1: {test_metrics.get('f1', 0.0):.4f}")

        # Save metrics to outputs/
        out_dir = Path(trainer.config['paths'].get('output_dir', 'outputs/'))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'test_metrics.yaml', 'w') as f:
            yaml.safe_dump(test_metrics, f, sort_keys=False)


if __name__ == '__main__':
    main()
