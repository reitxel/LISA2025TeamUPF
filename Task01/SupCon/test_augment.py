import os
import numpy as np
import torch
import nibabel as nib
from monai.data import Dataset, DataLoader
from train import prepare_data, get_transforms

# Configuration (adapt as needed)
TASK = "Noise"  # Change to desired task
SPATIAL_SIZE = (128, 128, 40)
BATCH_SIZE = 1
N_SAMPLES = 10
SEED = 42
OUTPUT_DIR = "example_aug"

# Set random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_tasks = [
    "Zipper", "Positioning", "Noise", "Banding",
    "Motion", "Contrast", "Distortion"
]

# Prepare data (using the same config as train.py)
train_files, _, _ = prepare_data(
    class_name=TASK,
    bids_dir="data/LISA2025/BIDS",
    qc_csv_path="data/LISA2025/BIDS/LISA_2025_bids.csv",
    augmented_dir="data/LISA2025/BIDS/derivatives/augmented_v3",
    augmented_qc_csv_path=(
        "data/LISA2025/BIDS/derivatives/augmented_v3/augmented_labels.csv"
    ),
    random_state=SEED
)

# Use training transforms (with augmentation)
train_transforms = get_transforms(SPATIAL_SIZE, stage="validation")
train_dataset = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Sample and save N_SAMPLES images
count = 0
for batch in train_loader:
    img = batch["img"][0].numpy()  # shape: (C, H, W, D) or (C, ...)
    label = batch["label"][0].item()
    print(batch["subject"][0])
    # Remove channel dimension if present
    if img.shape[0] == 1:
        img = img[0]
    # Save as NIfTI
    nii = nib.Nifti1Image(img, affine=np.eye(4))
    out_path = os.path.join(OUTPUT_DIR, f"aug_{count}_label{label}.nii.gz")
    nib.save(nii, out_path)
    print(f"Saved: {out_path}")
    count += 1
    if count >= N_SAMPLES:
        break

print(f"Done. {N_SAMPLES} augmented images saved to {OUTPUT_DIR}/") 