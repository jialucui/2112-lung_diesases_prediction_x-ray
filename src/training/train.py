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

from src.models.medical_models import create_model
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
            device=device
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
        
        # Gradient scaler
        self.scaler = GradScaler()
        
        # Metrics calculator
        self.metrics = MetricsCalculator()
        
        logger.info(f"Trainer initialized on {device}")
    
    def _compute_loss(self, binary_logits, severity_logits, batch):
        """Compute multi-task loss"""
        binary_labels = batch['label'].to(self.device)
        severity_labels = batch['severity'].to(self.device)
        
        binary_loss = nn.CrossEntropyLoss()(binary_logits, binary_labels)
        severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)
        
        binary_weight = self.config['training']['loss_weights']['binary_classification']
        severity_weight = self.config['training']['loss_weights']['severity_prediction']
        
        total_loss = binary_weight * binary_loss + severity_weight * severity_loss
        
        return total_loss, binary_loss, severity_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            
            with autocast():
                binary_logits, severity_logits = self.model(images)
                loss, _, _ = self._compute_loss(binary_logits, severity_logits, batch)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_max_norm']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1):.4f})
        
        return {'loss': total_loss / len(train_loader)}
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_binary_preds = []
        all_binary_targets = []
        all_severity_preds = []
        all_severity_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                
                with autocast():
                    binary_logits, severity_logits = self.model(images)
                    loss, _, _ = self._compute_loss(binary_logits, severity_logits, batch)
                
                total_loss += loss.item()
                
                binary_preds = binary_logits.argmax(dim=1).cpu()
                severity_preds = severity_logits.argmax(dim=1).cpu()
                
                all_binary_preds.extend(binary_preds.numpy())
                all_binary_targets.extend(batch['label'].numpy())
                all_severity_preds.extend(severity_preds.numpy())
                all_severity_targets.extend(batch['severity'].numpy())
        
        metrics = self.metrics.calculate_metrics(
            all_binary_preds,
            all_binary_targets,
            all_severity_preds,
            all_severity_targets
        )
        metrics['loss'] = total_loss / len(val_loader)
        
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
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    trainer = PneumoniaTrainer(args.config, device=args.device)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=trainer.config['data']['data_dir'],
        csv_file=trainer.config['data']['csv_file'],
        batch_size=trainer.config['training']['batch_size'],
        num_workers=trainer.config['data']['num_workers'],
        augment_train=trainer.config['data']['augment_train'],
        seed=trainer.config['data']['seed']
    )
    
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
