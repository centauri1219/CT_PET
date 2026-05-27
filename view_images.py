#!/usr/bin/env python3
"""
Script to view one CT image and 2 PET images side by side.
Usage: python view_images.py --ct <ct_path> --pet1 <pet1_path> --pet2 <pet2_path>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(path):
    """Load medical image from various formats."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        # Load numpy array
        img = np.load(path)
    elif ext == '.npz':
        # Load numpy compressed array
        data = np.load(path)
        # Try to get the first array
        img = data[data.files[0]]
    elif ext in ['.nii', '.gz']:
        # Load NIfTI format
        import nibabel as nib
        nii = nib.load(path)
        img = nii.get_fdata()
    elif ext in ['.dcm', '.dicom']:
        # Load DICOM format
        import pydicom
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
    else:
        # Try to load as numpy array by default
        try:
            img = np.load(path)
        except:
            raise ValueError(f"Unsupported file format: {ext}")
    
    return img


def get_middle_slice(img):
    """Get the middle slice of a 3D image, or return as-is if 2D."""
    if img.ndim == 3:
        # Return middle slice
        mid_slice = img.shape[2] // 2
        return img[:, :, mid_slice]
    elif img.ndim == 2:
        return img
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")


def normalize_for_display(img, percentile=99):
    """Normalize image for display using percentile scaling."""
    img_min = np.percentile(img, 100 - percentile)
    img_max = np.percentile(img, percentile)
    img_normalized = np.clip((img - img_min) / (img_max - img_min + 1e-8), 0, 1)
    return img_normalized


def view_images(ct_path, pet1_path, pet2_path, slice_idx=None):
    """
    Display CT and 2 PET images side by side.
    
    Args:
        ct_path: Path to CT image
        pet1_path: Path to first PET image
        pet2_path: Path to second PET image
        slice_idx: Specific slice index to display (None for middle slice)
    """
    # Load images
    print(f"Loading CT image from: {ct_path}")
    ct_img = load_image(ct_path)
    print(f"CT image shape: {ct_img.shape}")
    
    print(f"Loading PET1 image from: {pet1_path}")
    pet1_img = load_image(pet1_path)
    print(f"PET1 image shape: {pet1_img.shape}")
    
    print(f"Loading PET2 image from: {pet2_path}")
    pet2_img = load_image(pet2_path)
    print(f"PET2 image shape: {pet2_img.shape}")
    
    # Get slices
    if slice_idx is not None:
        if ct_img.ndim == 3:
            ct_slice = ct_img[:, :, slice_idx]
        else:
            ct_slice = ct_img
            
        if pet1_img.ndim == 3:
            pet1_slice = pet1_img[:, :, slice_idx]
        else:
            pet1_slice = pet1_img
            
        if pet2_img.ndim == 3:
            pet2_slice = pet2_img[:, :, slice_idx]
        else:
            pet2_slice = pet2_img
    else:
        ct_slice = get_middle_slice(ct_img)
        pet1_slice = get_middle_slice(pet1_img)
        pet2_slice = get_middle_slice(pet2_img)
    
    # Normalize for display
    ct_norm = normalize_for_display(ct_slice)
    pet1_norm = normalize_for_display(pet1_slice)
    pet2_norm = normalize_for_display(pet2_slice)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display CT
    axes[0].imshow(ct_norm, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    # Display PET1
    axes[1].imshow(pet1_norm, cmap='hot')
    axes[1].set_title('PET Image 1')
    axes[1].axis('off')
    
    # Display PET2
    axes[2].imshow(pet2_norm, cmap='hot')
    axes[2].set_title('PET Image 2')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='View one CT image and 2 PET images side by side'
    )
    parser.add_argument('--ct', type=str, required=True,
                        help='Path to CT image')
    parser.add_argument('--pet1', type=str, required=True,
                        help='Path to first PET image')
    parser.add_argument('--pet2', type=str, required=True,
                        help='Path to second PET image')
    parser.add_argument('--slice', type=int, default=None,
                        help='Specific slice index to display (default: middle slice)')
    
    args = parser.parse_args()
    
    # Check if files exist
    for path in [args.ct, args.pet1, args.pet2]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    view_images(args.ct, args.pet1, args.pet2, args.slice)


if __name__ == '__main__':
    main()
