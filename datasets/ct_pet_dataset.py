"""
CT-PET Dataset Loader with On-the-Fly Normalization
====================================================
Loads paired CT and PET scans from NIfTI files with proper normalization:
- CT: Window [-1000, 1000] HU and normalize to [-1, 1]
- PET: Log-transform SUV values and normalize to [-1, 1]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class CTPETDataset(Dataset):
    """
    Dataset for loading paired CT and PET 2D slices from NIfTI files.
    
    Expected directory structure:
        data_root/
            patient_001/
                A.nii.gz  (CT)
                B.nii.gz  (PET)
            patient_002/
                A.nii.gz
                B.nii.gz
            ...
    """
    
    def __init__(
        self,
        data_root: str,
        modality_ct: str = "A",
        modality_pet: str = "B",
        ct_window: Tuple[float, float] = (-1000, 1000),
        pet_log_scale: bool = True,
        pet_max_log_value: Optional[float] = None,
        transform=None
    ):
        """
        Args:
            data_root: Root directory containing patient folders
            modality_ct: CT modality name (default: "A")
            modality_pet: PET modality name (default: "B")
            ct_window: (min, max) for CT windowing in HU
            pet_log_scale: Whether to apply log-scaling to PET
            pet_max_log_value: Max log value for PET normalization (computed if None)
            transform: Optional data augmentation transforms
        """
        self.data_root = Path(data_root)
        self.modality_ct = modality_ct
        self.modality_pet = modality_pet
        self.ct_window = ct_window
        self.pet_log_scale = pet_log_scale
        self.pet_max_log_value = pet_max_log_value
        self.transform = transform
        
        # Find all patient directories
        self.patient_dirs = sorted([
            d for d in self.data_root.iterdir() 
            if d.is_dir() and self._has_required_files(d)
        ])
        
        if len(self.patient_dirs) == 0:
            raise ValueError(f"No valid patient directories found in {data_root}")
        
        print(f"Found {len(self.patient_dirs)} patients with paired CT-PET data")
        
        # Compute PET max log value if not provided
        if self.pet_log_scale and self.pet_max_log_value is None:
            print("Computing PET max log value from dataset...")
            self.pet_max_log_value = self._compute_pet_max_log()
            print(f"PET max log value: {self.pet_max_log_value:.4f}")
    
    def _has_required_files(self, patient_dir: Path) -> bool:
        """Check if patient directory has both CT and PET files."""
        ct_file = patient_dir / f"{self.modality_ct}.nii.gz"
        pet_file = patient_dir / f"{self.modality_pet}.nii.gz"
        return ct_file.exists() and pet_file.exists()
    
    def _compute_pet_max_log(self) -> float:
        """Compute maximum log-transformed PET value across all patients."""
        max_log = 0.0
        
        for patient_dir in self.patient_dirs[:min(20, len(self.patient_dirs))]:  # Sample 20 patients
            pet_file = patient_dir / f"{self.modality_pet}.nii.gz"
            pet_img = nib.load(str(pet_file))
            pet_data = pet_img.get_fdata()
            
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
        return len(self.patient_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and normalize a CT-PET pair.
        
        Returns:
            dict with:
                - ct: Normalized CT image [1, H, W]
                - pet: Normalized PET image [1, H, W]
                - patient_id: Patient identifier
        """
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name
        
        # Load CT
        ct_file = patient_dir / f"{self.modality_ct}.nii.gz"
        ct_img = nib.load(str(ct_file))
        ct_data = ct_img.get_fdata().squeeze()  # Remove extra dimensions
        
        # Load PET
        pet_file = patient_dir / f"{self.modality_pet}.nii.gz"
        pet_img = nib.load(str(pet_file))
        pet_data = pet_img.get_fdata().squeeze()
        
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
            'patient_id': patient_id
        }


def create_dataloaders(
    config: dict,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset
    dataset = CTPETDataset(
        data_root=config['data']['data_root'],
        modality_ct=config['data']['modality_ct'],
        modality_pet=config['data']['modality_pet'],
        ct_window=(config['data']['ct_window_min'], config['data']['ct_window_max']),
        pet_log_scale=config['data']['pet_log_scale'],
        pet_max_log_value=config['data'].get('pet_max_log_value', None)
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_vqvae']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_vqvae']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from pathlib import Path
    
    # Example config
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
        dataset = CTPETDataset(
            data_root=config['data']['data_root'],
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
        print(f"  Patient ID: {sample['patient_id']}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the data_root path is correct and contains patient folders with A.nii.gz and B.nii.gz files")
