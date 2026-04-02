"""
Medical Image Loading and Preprocessing

Handles:
- DICOM, JPG, PNG X-ray image formats
- Normalization (ImageNet or medical standards)
- Data augmentation
- Dataset creation and data loading
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pydicom
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.dcm', '.bmp', '.tif', '.tiff', '.webp')


def load_dicom(dicom_path: str) -> np.ndarray:
    """
    Load DICOM file and extract pixel array
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        Normalized pixel array as numpy array
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert to 3-channel if grayscale
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
        
        return pixel_array
    except Exception as e:
        logger.error(f"Error loading DICOM file {dicom_path}: {str(e)}")
        raise


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file (JPG, PNG, etc.)
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise


def get_image_statistics(image_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and std of dataset for normalization
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Tuple of (mean, std) for normalization
    """
    logger.info(f"Calculating image statistics for {image_dir}...")
    
    images = []
    image_dir = str(image_dir)

    # Walk recursively to support folder-based datasets
    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.dcm'):
                    img = load_dicom(file_path)
                else:
                    img = load_image(file_path)

                # Resize to standard size
                img = cv2.resize(img, (224, 224))
                images.append(img)
            except Exception:
                continue
    
    if not images:
        logger.warning("No images found. Using default ImageNet statistics.")
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    images = np.array(images) / 255.0
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    
    logger.info(f"Mean: {mean}, Std: {std}")
    return mean, std


class XrayDataset(Dataset):
    """
    Custom PyTorch Dataset for chest X-ray images
    
    Supports:
    - Binary classification (Normal/Pneumonia)
    - Multi-task learning (Binary classification + Severity)
    - Data augmentation
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 severity_labels: Optional[List[int]] = None,
                 image_size: int = 224,
                 augment: bool = False,
                 normalize: bool = True,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of binary labels (0=Normal, 1=Pneumonia)
            severity_labels: List of severity labels (0=Mild, 1=Moderate, 2=Severe)
            image_size: Size to resize images to
            augment: Whether to apply augmentation
            normalize: Whether to normalize images
            mean: Mean values for normalization
            std: Std values for normalization
        """
        self.image_paths = image_paths
        self.labels = labels
        self.severity_labels = severity_labels
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Use ImageNet statistics if not provided
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406])
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225])
        
        # Define transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std) if normalize else transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std) if normalize else transforms.ToTensor(),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item from dataset
        
        Returns:
            Dict with 'image' and 'label' keys, optionally 'severity' key
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        if image_path.lower().endswith('.dcm'):
            image = load_dicom(image_path)
        else:
            image = load_image(image_path)
        
        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert BGR to RGB for PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Apply transforms
        image = self.transform(image)
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': image_path
        }
        
        # Add severity label if available
        if self.severity_labels is not None:
            severity = self.severity_labels[idx]
            sample['severity'] = torch.tensor(severity, dtype=torch.long)
        
        return sample


def create_data_loaders(
    data_dir: str,
    csv_file: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    augment_train: bool = True,
    seed: int = 42,
    severity_strategy: str = "none",
    synthetic_severity_by_class: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders
    
    Args:
        data_dir: Directory containing images
        csv_file: CSV file with columns: image_name, label, [severity]
        batch_size: Batch size for dataloaders
        image_size: Image size after resizing
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        num_workers: Number of worker processes
        augment_train: Whether to augment training data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = str(data_dir)

    image_paths: List[str] = []
    labels: np.ndarray
    severity_labels: Optional[np.ndarray] = None

    # Option A: CSV-based dataset (legacy)
    if csv_file and os.path.exists(str(csv_file)):
        import pandas as pd

        logger.info(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        # Prepare paths and labels
        image_paths = [os.path.join(data_dir, img) for img in df['image_name']]
        labels = df['label'].values
        severity_labels = df['severity'].values if 'severity' in df.columns else None

    # Option B: folder-based dataset (recommended)
    else:
        # Use immediate subfolders of data_dir as class names
        class_dirs = [
            p for p in sorted(Path(data_dir).iterdir())
            if p.is_dir() and not p.name.startswith('.')
        ]

        if len(class_dirs) < 2:
            raise ValueError(
                f"Folder-based dataset expects at least 2 class subfolders under: {data_dir}\n"
                f"Found: {[p.name for p in class_dirs]}"
            )

        class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}
        logger.info(f"Detected classes: {class_to_idx}")

        tmp_labels: List[int] = []
        for class_dir in class_dirs:
            for root, _, files in os.walk(str(class_dir)):
                for file in files:
                    if not file.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    image_paths.append(os.path.join(root, file))
                    tmp_labels.append(class_to_idx[class_dir.name])

        labels = np.array(tmp_labels, dtype=np.int64)

        if csv_file is None or not os.path.exists(str(csv_file)):
            if severity_strategy == "synthetic":
                if not synthetic_severity_by_class:
                    raise ValueError(
                        "severity_strategy='synthetic' requires synthetic_severity_by_class "
                        "(one severity class id per label index)."
                    )
                n_cls = int(labels.max()) + 1 if len(labels) else 0
                if len(synthetic_severity_by_class) < n_cls:
                    raise ValueError(
                        f"synthetic_severity_by_class length {len(synthetic_severity_by_class)} "
                        f"< num classes {n_cls}"
                    )
                severity_labels = np.array(
                    [synthetic_severity_by_class[int(lab)] for lab in labels],
                    dtype=np.int64,
                )
                logger.info(
                    "Using synthetic severity labels from synthetic_severity_by_class "
                    f"(strategy={severity_strategy})."
                )
    
    # Filter valid paths
    valid_indices = [i for i, p in enumerate(image_paths) if os.path.exists(p)]
    image_paths = [image_paths[i] for i in valid_indices]
    labels = labels[valid_indices]
    if severity_labels is not None:
        severity_labels = severity_labels[valid_indices]
    
    logger.info(f"Found {len(image_paths)} valid images")
    
    # Calculate statistics
    mean, std = get_image_statistics(data_dir)
    
    # Create dataset
    dataset = XrayDataset(
        image_paths=image_paths,
        labels=labels,
        severity_labels=severity_labels,
        image_size=image_size,
        augment=False,  # Will apply augmentation in loaders
        normalize=True,
        mean=mean,
        std=std
    )
    
    # Split dataset
    if (train_split + val_split + test_split) <= 0:
        raise ValueError("train_split + val_split + test_split must be > 0")

    total = train_split + val_split + test_split
    train_split_n = train_split / total
    val_split_n = val_split / total
    test_split_n = test_split / total

    train_size = int(len(dataset) * train_split_n)
    val_size = int(len(dataset) * val_split_n)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply augmentation to train dataset
    train_dataset.dataset.augment = augment_train
    
    logger.info(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if test_size > 0 else None

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing data loading utilities...")
    print("✅ Module ready for use")