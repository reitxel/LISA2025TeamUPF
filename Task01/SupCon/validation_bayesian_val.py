"""
Validation script with Bayesian Network hybrid inference for LISA 2025 QC data.
This script loads trained models, trains a Bayesian Network on the full training 
set, and performs inference on validation images with BN adjustment.
"""

import os
import argparse
import glob
import numpy as np
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
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
import pickle

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


def load_models(model_dir, model_type, device, tasks):
    """
    Load all models for the specified tasks.
    
    Args:
        model_dir: Directory containing model checkpoints
        model_type: Type of model ('contrastive' or 'focalemd')
        device: Device to load models on
        tasks: List of task names
        
    Returns:
        dict: Dictionary mapping task names to loaded models
    """
    model_class = (
        OrdinalContrastiveLightningModule
        if model_type == "contrastive" else OrdinalFocalEMDLightningModule
    )
    models = {}
    for task in tasks:
        model_path = os.path.join(model_dir, f"{task}_finalmodel.ckpt")
        if os.path.exists(model_path):
            print(f"Loading model for {task}...")
            models[task] = (
                model_class.load_from_checkpoint(model_path)
                .to(device)
            )
            models[task].eval()
        else:
            print(
                f"Warning: Model not found for {task} at {model_path}. "
                f"Skipping."
            )
    return models


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


def predict_image_probabilities(
    models, image_path, transforms, device, tasks
):
    """
    Get probability predictions for all tasks for a single image.
    
    Args:
        models: Dictionary of loaded models
        image_path: Path to the image file
        transforms: Data transforms to apply
        device: Device to run inference on
        tasks: List of task names
        
    Returns:
        np.ndarray: Probability array of shape (num_tasks, 3)
    """
    data_dict = {"img": image_path}
    processed = transforms(data_dict)
    image_tensor = processed["img"].unsqueeze(0).to(device)
    probs = np.zeros((len(tasks), 3), dtype=np.float32)
    
    for i, task in enumerate(tasks):
        if task in models:
            with torch.no_grad():
                logits = models[task](image_tensor)
                softmax = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs[i, :] = softmax
        else:
            probs[i, :] = np.nan
    
    return probs


def run_bn_soft_evidence_v2(
    bn, cpts, tasks, orig_probs, epsilon=1e-8, 
    confidence_threshold=0.6
):
    """
    Alternative using hard evidence from confident DL predictions.
    
    Args:
        bn: BayesianNetwork object
        cpts: Not used (kept for compatibility)
        tasks: List of task names
        orig_probs: Original DL probabilities, shape (7,3)
        epsilon: Numerical stability
        confidence_threshold: Minimum confidence to use as evidence
    
    Returns:
        adjusted_probs: Adjusted probabilities, shape (7,3)
    """
    from pgmpy.inference import VariableElimination
    
    adjusted_probs = orig_probs.copy()
    
    try:
        inference = VariableElimination(bn)
        
        # For each task, use confident predictions from other tasks as evidence
        for i, target_task in enumerate(tasks):
            try:
                # Collect evidence from other confident predictions
                evidence = {}
                for j, other_task in enumerate(tasks):
                    if i != j:  # Don't use self as evidence
                        max_prob = np.max(orig_probs[j])
                        if max_prob > confidence_threshold:
                            predicted_class = np.argmax(orig_probs[j])
                            evidence[other_task] = predicted_class
                
                if evidence:
                    # Query with evidence from other tasks
                    result = inference.query(
                        variables=[target_task], evidence=evidence
                    )
                    bn_conditional = result.values
                else:
                    # No confident evidence, use marginal
                    result = inference.query(variables=[target_task])
                    bn_conditional = result.values
                
                # Simple mixing
                alpha = 0.3  # Fixed mixing weight
                combined = (1 - alpha) * orig_probs[i] + alpha * bn_conditional
                combined = np.clip(combined, epsilon, 1.0)
                adjusted_probs[i] = combined / combined.sum()
                
            except Exception as e:
                print(f"BN inference error for {target_task}: {e}")
                adjusted_probs[i] = orig_probs[i]
                
    except Exception as e:
        print(f"BN initialization error: {e}")
        return orig_probs
    
    return adjusted_probs


def train_bayesian_network(train_csv_path, tasks):
    """
    Train a Bayesian Network on the full training set.
    
    Args:
        train_csv_path: Path to training CSV with labels
        tasks: List of task names
        
    Returns:
        DiscreteBayesianNetwork: Trained Bayesian Network
    """
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[train_df['filename'].notna()]
    
    # Extract labels for training
    train_labels = train_df[tasks].fillna(0).astype(int)
    
    # Build fully connected DAG (all possible edges in topological order)
    edges = []
    for i, parent in enumerate(tasks):
        for child in tasks[i+1:]:
            edges.append((parent, child))
    
    bn = DiscreteBayesianNetwork(edges)
    
    # Add all nodes explicitly (in case of no edges)
    for node in tasks:
        if node not in bn.nodes:
            bn.add_node(node)
    
    # Learn CPTs
    bn.fit(train_labels, estimator=MaximumLikelihoodEstimator)
    
    return bn


def main():
    """Main validation function with Bayesian Network adjustment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on images using trained models with BN "
                    "adjustment"
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
        "--train_csv",
        type=str,
        default="data/LISA2025/BIDS_norm/LISA_2025_bids.csv",
        help="CSV file with training data for BN training"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_emd_augmented_both_final/LISA_LF_QC_predictions_bn.csv",
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
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for numerical stability in probabilities"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for BN evidence"
    )
    parser.add_argument(
        "--save_bn",
        type=str,
        default=None,
        help="Path to save the trained Bayesian Network (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    
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
    models = load_models(args.model_dir, args.model_type, args.device, tasks)
    
    if not models:
        raise ValueError("No valid models found in the model directory")
    
    # Train Bayesian Network on full training set
    print("Training Bayesian Network on full training set...")
    bn = train_bayesian_network(args.train_csv, tasks)
    
    # Save BN if requested
    if args.save_bn:
        os.makedirs(os.path.dirname(args.save_bn), exist_ok=True)
        with open(args.save_bn, "wb") as f:
            pickle.dump(bn, f)
        print(f"Saved Bayesian Network to {args.save_bn}")
    
    # Process each image
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: "
              f"{os.path.basename(image_path)}")
        
        # Initialize result dictionary
        result = {"filename": os.path.basename(image_path)}
        
        try:
            # Get original probabilities from all models
            orig_probs = predict_image_probabilities(
                models, image_path, transforms, args.device, tasks
            )
            
            # Apply Bayesian Network adjustment
            adj_probs = run_bn_soft_evidence_v2(
                bn, None, tasks, orig_probs, 
                epsilon=args.epsilon,
                confidence_threshold=args.confidence_threshold
            )
            
            # Check if adjusted probabilities have NaN values
            if np.isnan(adj_probs).any() or np.isnan(adj_probs).all():
                print(f"Warning: Adjusted probabilities contain NaN for {os.path.basename(image_path)}, using original probabilities")
                adj_probs = orig_probs.copy()
            
            # Get predictions from adjusted probabilities
            adj_pred = np.nanargmax(adj_probs, axis=1)
            
            # Store results
            for j, task in enumerate(tasks):
                result[task] = adj_pred[j]  # Use BN-adjusted predictions
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Try to get original probabilities even if adjustment fails
            try:
                orig_probs = predict_image_probabilities(
                    models, image_path, transforms, args.device, tasks
                )
                orig_pred = np.nanargmax(orig_probs, axis=1)
                # Use original predictions as fallback
                for j, task in enumerate(tasks):
                    result[task] = orig_pred[j]
            except Exception as e2:
                print(f"Failed to get original probabilities for {os.path.basename(image_path)}: {e2}")
                # Fill with None if processing failed completely
                for task in tasks:
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

    # sort by filename
    df.sort_values("filename", inplace=True)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(df)} images")


if __name__ == "__main__":
    main() 