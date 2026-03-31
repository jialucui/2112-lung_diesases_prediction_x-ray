"""
Complete training loop for pneumonia detection model with:
- Multi-task learning (classification + severity prediction)
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- TensorBoard logging
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import yaml
from pathlib import Path

from ..models.medical_models import DenseNetMultiTask
from ..preprocessing.dicom_xray_loader import ChestXrayDataset


class TrainingConfig:
    """Training configuration manager"""
    
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.setup_paths()
    
    def setup_paths(self):
        """Create necessary directories"""
        for path_key in ['checkpoint_dir', 'log_dir', 'output_dir']:
            path = self.config['paths'][path_key]
            Path(path).mkdir(parents=True, exist_ok=True)


class MetricTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name, last_n=None):
        if metric_name not in self.metrics:
            return 0
        values = self.metrics[metric_name]
        if last_n:
            values = values[-last_n:]
        return np.mean(values) if values else 0
    
    def reset(self):
        self.metrics = {}


class EarlyStoppingCallback:
    """Early stopping callback"""
    
    def __init__(self, patience=10, metric_name='val_loss', save_best=True):
        self.patience = patience
        self.metric_name = metric_name
        self.save_best = save_best
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
    
    def __call__(self, current_value, epoch):
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                return True
        return False


class ModelTrainer:
    """Main trainer class"""
    
    def __init__(self, config_path='configs/config.yaml', device='cuda'):
        self.config = TrainingConfig(config_path).config
        self.device = device
        
        # Initialize model
        self.model = DenseNetMultiTask(
            num_classes=2,
            severity_classes=4,
            pretrained=self.config['model']['pretrained']
        ).to(device)
        
        # Loss functions
        self.loss_classification = nn.CrossEntropyLoss()
        self.loss_severity = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        # Mixed precision
        self.use_amp = self.config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics
        self.metric_tracker = MetricTracker()
        self.early_stopping = EarlyStoppingCallback(
            patience=self.config['training']['early_stopping_patience'],
            metric_name='val_loss'
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        self.metric_tracker.reset()
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (images, labels, severity) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            severity = severity.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Forward pass
                class_logits, severity_logits = self.model(images)
                
                # Compute losses
                loss_cls = self.loss_classification(class_logits, labels)
                loss_sev = self.loss_severity(severity_logits, severity)
                
                # Multi-task loss
                weights = self.config['training']['loss_weights']
                loss = (weights['binary_classification'] * loss_cls + 
                       weights['severity_prediction'] * loss_sev)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_max_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_max_norm']
                )
                self.optimizer.step()
            
            # Track metrics
            self.metric_tracker.update('train_loss', loss.item())
            self.metric_tracker.update('train_loss_cls', loss_cls.item())
            self.metric_tracker.update('train_loss_sev', loss_sev.item())
            
            progress_bar.set_postfix({
                'loss': self.metric_tracker.get_average('train_loss', last_n=10),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return self.metric_tracker.get_average('train_loss')
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        self.metric_tracker.reset()
        
        progress_bar = tqdm(val_loader, desc='Validating')
        
        for images, labels, severity in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            severity = severity.to(self.device)
            
            with autocast(enabled=self.use_amp):
                class_logits, severity_logits = self.model(images)
                loss_cls = self.loss_classification(class_logits, labels)
                loss_sev = self.loss_severity(severity_logits, severity)
                
                weights = self.config['training']['loss_weights']
                loss = (weights['binary_classification'] * loss_cls + 
                       weights['severity_prediction'] * loss_sev)
            
            self.metric_tracker.update('val_loss', loss.item())
            self.metric_tracker.update('val_loss_cls', loss_cls.item())
            self.metric_tracker.update('val_loss_sev', loss_sev.item())
        
        return self.metric_tracker.get_average('val_loss')
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        
        best_checkpoint = os.path.join(
            self.config['paths']['checkpoint_dir'],
            'best_model.pth'
        )
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best checkpoint
            if epoch == 0 or val_loss < self.early_stopping.best_value:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, best_checkpoint)
                print(f"Saved best checkpoint to {best_checkpoint}")
            
            # Early stopping
            if self.early_stopping(val_loss, epoch):
                print(f"Training stopped early at epoch {epoch}")
                break
        
        print("Training completed!")
        return best_checkpoint


def train_model(config_path='configs/config.yaml', 
                train_csv='data/metadata.csv',
                device='cuda'):
    """
    Main training function
    
    Args:
        config_path: Path to config file
        train_csv: Path to training metadata
        device: Device to use (cuda/cpu)
    """
    
    # Initialize trainer
    trainer = ModelTrainer(config_path, device)
    
    # Load datasets
    train_dataset = ChestXrayDataset(
        csv_file=train_csv,
        split='train',
        image_size=trainer.config['data']['image_size'],
        augment=trainer.config['data']['augment_train']
    )
    
    val_dataset = ChestXrayDataset(
        csv_file=train_csv,
        split='val',
        image_size=trainer.config['data']['image_size'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer.config['training']['batch_size'],
        shuffle=True,
        num_workers=trainer.config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer.config['training']['batch_size'],
        shuffle=False,
        num_workers=trainer.config['data']['num_workers']
    )
    
    # Train
    best_model_path = trainer.train(train_loader, val_loader)
    
    return best_model_path


if __name__ == '__main__':
    train_model()
