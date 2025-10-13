"""
Validation script for the Ordinal Contrastive DenseNet on LISA 2025 QC data.
This script loads trained models and performs inference on new images.
"""

import os
import argparse
import glob

import pandas as pd
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)

# Import our custom modules
from model import OrdinalContrastiveLightningModule
from model import OrdinalFocalEMDLightningModule


def get_inference_transforms(spatial_size: tuple = (128, 128, 128)):
    """
    Get data transforms for inference (no augmentation).
    
    Args:
        spatial_size: Spatial size for cropping and padding
        
    Returns:
        Compose: Transform pipeline for inference
    """
    transforms = [
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(
            keys=["img"], method="symmetric", spatial_size=spatial_size
        ),
        ToTensord(keys=["img"], dtype=torch.float32),
    ]
    
    return Compose(transforms)


def load_model(
    model_path: str, model_class, device: str = "cuda"
):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        model_class: Model class to instantiate
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model from checkpoint
    model = model_class.load_from_checkpoint(model_path)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def predict_single_image(
    model: OrdinalContrastiveLightningModule,
    image_path: str,
    transforms,
    device: str = "cuda"
) -> int:
    """
    Perform inference on a single image.
    
    Args:
        model: Loaded model
        image_path: Path to the image file
        transforms: Data transforms to apply
        device: Device to run inference on
        
    Returns:
        int: Predicted class (0, 1, or 2)
    """
    # Prepare data dictionary
    data_dict = {"img": image_path}
    
    # Apply transforms
    processed_data = transforms(data_dict)
    # Add batch dimension
    image_tensor = processed_data["img"].unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        # Use the model's forward method which returns classification logits
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    return prediction


def main():
    """Main validation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on images using trained models"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/LISA2025/Task_1_Validation/",
        help="Directory containing .nii.gz files"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results_emd_augmented_both_final/",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_emd_augmented_both_final/LISA_LF_QC_predictions.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=[150, 150, 150],
        help="Spatial size for image processing"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["contrastive", "focalemd"],
        default="focalemd",
        help="Type of model to use: 'contrastive' or 'focalemd'"
    )
    
    args = parser.parse_args()
    
    if args.model_type == "contrastive":
        model_class = OrdinalContrastiveLightningModule
    elif args.model_type == "focalemd":
        model_class = OrdinalFocalEMDLightningModule
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Define tasks (same as in training)
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    
    # Get all .nii.gz files in the input directory
    image_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    if not image_files:
        raise ValueError(f"No .nii.gz files found in {args.input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Set up transforms
    transforms = get_inference_transforms(tuple(args.spatial_size))
    
    # Load models for each task
    models = {}
    for task in tasks:
        model_path = os.path.join(args.model_dir, f"{task}_finalmodel.ckpt")
        if os.path.exists(model_path):
            print(f"Loading model for {task}...")
            models[task] = load_model(model_path, model_class, args.device)
        else:
            print(f"Warning: Model not found for {task} at {model_path}")
            print(f"Skipping {task}...")
    
    if not models:
        raise ValueError("No valid models found in the model directory")
    
    # Process each image
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: "
              f"{os.path.basename(image_path)}")
        
        # Initialize result dictionary
        result = {"filename": os.path.basename(image_path)}
        
        # Run inference for each task
        for task, model in models.items():
            try:
                prediction = predict_single_image(
                    model, image_path, transforms, args.device
                )
                result[task] = prediction
            except Exception as e:
                print(f"Error processing {task} for {image_path}: {str(e)}")
                result[task] = None
        
        results.append(result)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Ensure all task columns exist (fill with None if missing)
    for task in tasks:
        if task not in df.columns:
            df[task] = None
    
    # Reorder columns to have filename first, then tasks
    column_order = ["filename"] + tasks
    df = df[column_order]

    # Sort by filename
    df.sort_values("filename", inplace=True)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(df)} images")


if __name__ == "__main__":
    main() 