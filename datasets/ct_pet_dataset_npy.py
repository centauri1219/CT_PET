"""
CT-PET Dataset Loader for NPY Files
====================================
Loads paired CT and PET 2D slices from numpy files with proper normalization:
- CT: Window [-1000, 1000] HU and normalize to [-1, 1]
- PET: Log-transform SUV values and normalize to [-1, 1]

Expected directory structure:
    data_root/
        train/
            A/  (CT .npy files)
            B/  (PET .npy files)
        val/
            A/
            B/
        test/
            A/
            B/
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class CTPETDatasetNPY(Dataset):
    """
    Dataset for loading paired CT and PET 2D slices from numpy files.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        modality_ct: str = "A",
        modality_pet: str = "B",
        ct_window: Tuple[float, float] = (-1000, 1000),
        pet_log_scale: bool = True,
        pet_max_log_value: Optional[float] = None,
        transform=None
    ):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: Dataset split ('train', 'val', or 'test')
            modality_ct: CT modality folder name (default: "A")
            modality_pet: PET modality folder name (default: "B")
            ct_window: (min, max) for CT windowing in HU
            pet_log_scale: Whether to apply log-scaling to PET
            pet_max_log_value: Max log value for PET normalization (computed if None)
            transform: Optional data augmentation transforms
        """
        self.data_root = Path(data_root)
        self.split = split
        self.modality_ct = modality_ct
        self.modality_pet = modality_pet
        self.ct_window = ct_window
        self.pet_log_scale = pet_log_scale
        self.pet_max_log_value = pet_max_log_value
        self.transform = transform
        
        # Get paths to CT and PET directories
        self.ct_dir = self.data_root / split / modality_ct
        self.pet_dir = self.data_root / split / modality_pet
        
        if not self.ct_dir.exists():
            raise ValueError(f"CT directory not found: {self.ct_dir}")
        if not self.pet_dir.exists():
            raise ValueError(f"PET directory not found: {self.pet_dir}")
        
        # Get list of files
        self.ct_files = sorted([f for f in self.ct_dir.glob('*.npy')])
        self.pet_files = sorted([f for f in self.pet_dir.glob('*.npy')])
        
        if len(self.ct_files) == 0:
            raise ValueError(f"No .npy files found in {self.ct_dir}")
        if len(self.pet_files) == 0:
            raise ValueError(f"No .npy files found in {self.pet_dir}")
        
        # Match CT and PET files by name
        self.paired_files = self._match_files()
        
        print(f"[{split}] Found {len(self.paired_files)} paired CT-PET slices")
        
        # Compute PET max log value if not provided
        if self.pet_log_scale and self.pet_max_log_value is None:
            print(f"[{split}] Computing PET max log value from dataset...")
            self.pet_max_log_value = self._compute_pet_max_log()
            print(f"[{split}] PET max log value: {self.pet_max_log_value:.4f}")
    
    def _match_files(self):
        """Match CT and PET files by name."""
        ct_names = {f.name: f for f in self.ct_files}
        pet_names = {f.name: f for f in self.pet_files}
        
        # Find common names
        common_names = set(ct_names.keys()) & set(pet_names.keys())
        
        if len(common_names) == 0:
            raise ValueError("No matching CT-PET pairs found!")
        
        paired = [(ct_names[name], pet_names[name]) for name in sorted(common_names)]
        return paired
    
    def _compute_pet_max_log(self) -> float:
        """Compute maximum log-transformed PET value across sample of data."""
        max_log = 0.0
        sample_size = min(100, len(self.paired_files))
        
        for i in range(0, len(self.paired_files), len(self.paired_files) // sample_size):
            _, pet_file = self.paired_files[i]
            pet_data = np.load(pet_file)
            
            # Apply log transform
            pet_log = np.log(1 + pet_data)
            max_log = max(max_log, np.max(pet_log))
        
        return max_log
    
    def normalize_ct(self, ct_data: np.ndarray) -> np.ndarray:
        """
        Normalize CT data with windowing.
        
        Formula: x_norm = (clip(x, min, max) - min) / (max - min) * 2 - 1
        Result: Range [-1, 1]
        """
        ct_min, ct_max = self.ct_window
        
        # Clip to window
        ct_clipped = np.clip(ct_data, ct_min, ct_max)
        
        # Normalize to [-1, 1]
        ct_norm = (ct_clipped - ct_min) / (ct_max - ct_min) * 2 - 1
        
        return ct_norm.astype(np.float32)
    
    def normalize_pet(self, pet_data: np.ndarray) -> np.ndarray:
        """
        Normalize PET data with log-scaling.
        
        Formula: x_log = log(1 + x)
                x_norm = (x_log / max_log) * 2 - 1
        Result: Range [-1, 1]
        """
        if self.pet_log_scale:
            # Apply log transform
            pet_log = np.log(1 + pet_data)
            
            # Normalize to [-1, 1]
            pet_norm = (pet_log / self.pet_max_log_value) * 2 - 1
        else:
            # Simple min-max normalization (not recommended)
            pet_max = np.max(pet_data)
            if pet_max > 0:
                pet_norm = (pet_data / pet_max) * 2 - 1
            else:
                pet_norm = pet_data
        
        return pet_norm.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.paired_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and normalize a CT-PET pair.
        
        Returns:
            dict with:
                - ct: Normalized CT image [1, H, W]
                - pet: Normalized PET image [1, H, W]
                - slice_id: Slice identifier
        """
        ct_file, pet_file = self.paired_files[idx]
        slice_id = ct_file.stem
        
        # Load data
        ct_data = np.load(ct_file)
        pet_data = np.load(pet_file)
        
        # Normalize
        ct_norm = self.normalize_ct(ct_data)
        pet_norm = self.normalize_pet(pet_data)
        
        # Convert to tensors [1, H, W]
        ct_tensor = torch.from_numpy(ct_norm).unsqueeze(0)
        pet_tensor = torch.from_numpy(pet_norm).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            # Stack for joint transformation
            combined = torch.cat([ct_tensor, pet_tensor], dim=0)
            combined = self.transform(combined)
            ct_tensor, pet_tensor = combined[0:1], combined[1:2]
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'patient_id': slice_id
        }


def create_dataloaders_npy(
    config: dict,
    batch_size: int = None,
    num_workers: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for NPY files.
    
    Args:
        config: Configuration dictionary
        batch_size: Override batch size from config
        num_workers: Override num_workers from config
    
    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = batch_size or config['train_vqvae']['batch_size']
    num_workers = num_workers or config['data']['num_workers']
    
    # Create datasets for each split
    train_dataset = CTPETDatasetNPY(
        data_root=config['data']['data_root'],
        split='train',
        modality_ct=config['data']['modality_ct'],
        modality_pet=config['data']['modality_pet'],
        ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
        pet_log_scale=config['data']['pet_log_scale'],
        pet_max_log_value=config['data'].get('pet_max_log_value', None)
    )
    
    val_dataset = CTPETDatasetNPY(
        data_root=config['data']['data_root'],
        split='val',
        modality_ct=config['data']['modality_ct'],
        modality_pet=config['data']['modality_pet'],
        ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
        pet_log_scale=config['data']['pet_log_scale'],
        pet_max_log_value=train_dataset.pet_max_log_value  # Use same as train
    )
    
    test_dataset = CTPETDatasetNPY(
        data_root=config['data']['data_root'],
        split='test',
        modality_ct=config['data']['modality_ct'],
        modality_pet=config['data']['modality_pet'],
        ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
        pet_log_scale=config['data']['pet_log_scale'],
        pet_max_log_value=train_dataset.pet_max_log_value  # Use same as train
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    config = {
        'data': {
            'data_root': '/scratch/b24cs1085/CPDM/processed_data',
            'modality_ct': 'A',
            'modality_pet': 'B',
            'ct_window_min': -1000,
            'ct_window_max': 1000,
            'pet_log_scale': True,
            'pet_max_log_value': None,
            'num_workers': 4
        },
        'train_vqvae': {
            'batch_size': 4
        },
        'inference': {
            'batch_size': 4
        }
    }
    
    try:
        dataset = CTPETDatasetNPY(
            data_root=config['data']['data_root'],
            split='train',
            modality_ct=config['data']['modality_ct'],
            modality_pet=config['data']['modality_pet'],
            ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
            pet_log_scale=config['data']['pet_log_scale']
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"\nSample loaded:")
        print(f"  CT shape: {sample['ct'].shape}")
        print(f"  PET shape: {sample['pet'].shape}")
        print(f"  CT range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
        print(f"  PET range: [{sample['pet'].min():.3f}, {sample['pet'].max():.3f}]")
        print(f"  Slice ID: {sample['patient_id']}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
