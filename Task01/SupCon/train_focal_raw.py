"""
Training script for the Ordinal Contrastive DenseNet on LISA 2025 QC data.
This script demonstrates how to use the combined loss function that adapts
both SCOL and binary imbalance handling strategies.
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
    RandAffined,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    Rand3DElasticd,
    RandFlipd,
)
from sklearn.model_selection import StratifiedGroupKFold


from model import OrdinalFocalEMDLightningModule
from model import plot_confusion_matrix


def prepare_data(
    class_name: str,
    bids_dir: str,
    qc_csv_path: str,
    augmented_dir: str,
    augmented_qc_csv_path: str,
    random_state: int = 42,
    use_raw_only: bool = True
) -> Tuple[List[Dict], List[Dict], torch.Tensor]:
    """
    Prepare data for training, handling the ordinal nature and class imbalance.
    """
    # Load raw and augmented data
    qc_df = pd.read_csv(qc_csv_path)
    qc_df["data_type"] = "raw"
    print("Raw samples: ", len(qc_df))
    aug_qc_df = pd.read_csv(augmented_qc_csv_path)
    aug_qc_df["data_type"] = "augmented"
    print("Augmented samples: ", len(aug_qc_df))
    
    # Combine dataframes
    combined_df = pd.concat([qc_df, aug_qc_df], ignore_index=True)
    print("Combined samples: ", len(combined_df))
    combined_df = combined_df[combined_df['filename'].notna()]
    print("Combined samples after filtering: ", len(combined_df))

    # Use only raw data if specified
    if use_raw_only:
        combined_df = combined_df[combined_df["data_type"] == "raw"]
        print(
            "Combined samples after filtering for raw only: ",
            len(combined_df)
        )
    else:
        print("Using both raw and augmented data: ", len(combined_df))
        
    # Get all task columns
    task_columns = [
        "Noise", "Zipper", "Positioning", "Banding", 
        "Motion", "Contrast", "Distortion"
    ]
    
    # Filter for specific task (single-task learning, allow at most one other 
    # task labeled as 1 or 2)
    other_tasks = [col for col in task_columns if col != class_name]
    other_labeled_count = (combined_df[other_tasks].isin([1, 2])).sum(axis=1)
    other_task_mask = other_labeled_count <= 1
    
    # Apply the single-task learning filter
    combined_df = combined_df[other_task_mask]
    print(f"After single-task filtering: {len(combined_df)} samples")

    # For class 0, keep all raw and 50% of augmented; for class 1/2, keep all
    label_col = class_name
    is_class_0 = combined_df[label_col] == 0
    is_raw = combined_df["data_type"] == "raw"
    is_aug = combined_df["data_type"] == "augmented"
    is_class_1_2 = combined_df[label_col].isin([1, 2])

    # Class 0, raw: keep all
    class0_raw = combined_df[is_class_0 & is_raw]
    # Class 0, augmented: sample 50% with random_state, only if we are not using raw only
    if not use_raw_only:
        class0_aug = combined_df[is_class_0 & is_aug].sample(
            frac=0.5, random_state=random_state
        )
    else:
        class0_aug = pd.DataFrame()
    # Class 1/2: keep all
    class1_2 = combined_df[is_class_1_2]

    # Combine all
    combined_df = pd.concat(
        [class0_raw, class0_aug, class1_2], ignore_index=True
    )
    print(f"Number of samples for {class_name}: {len(combined_df)}")
    print(combined_df.head())
    # Print columns
    print(combined_df.columns)

    # Extract data
    filenames = combined_df['filename'].astype(str).values
    labels = combined_df[class_name].values.astype(int)

    # subjects is the first part of the basename of the filenames,
    # being sub-XXXX
    subjects = [
        os.path.basename(filename).split('_')[0]
        for filename in filenames
    ]
        
    # Create stratified splits ensuring no data leakage between subjects
    sgkf = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    
    # Get first fold for train/val split
    train_idx, val_idx = next(sgkf.split(filenames, labels, subjects))
    
    # Create data dictionaries
    train_files = [
        {"img": filenames[i], "label": labels[i], "subject": subjects[i]} 
        for i in train_idx
    ]
    val_files = [
        {"img": filenames[i], "label": labels[i], "subject": subjects[i]} 
        for i in val_idx
    ]

    # Calculate class weights based on training set only
    train_labels = labels[train_idx]
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    class_weights = len(train_labels) / (len(unique_labels) * label_counts)
    # Ensure class_weights is length 3 (for 3 classes)
    full_class_weights = np.zeros(3, dtype=np.float32)
    for label, weight in zip(unique_labels, class_weights):
        full_class_weights[label] = weight
    class_weights = torch.tensor(full_class_weights, dtype=torch.float32)
    
    print(f"\nClass distribution for {class_name}:")
    for label, count, weight in zip(unique_labels, label_counts,
                                    class_weights):
        print(f"Class {label}: {count} samples (weight: {weight:.3f})")
    
    # Print the amount and label distribution of the train and val sets
    print(f"\nTrain set distribution for {class_name}:")
    train_df = pd.DataFrame(train_files)
    print(train_df['label'].value_counts())
    print(f"\nVal set distribution for {class_name}:")
    val_df = pd.DataFrame(val_files)
    print(val_df['label'].value_counts())
    
    # Use the full, unbalanced validation set
    balanced_val = val_df.to_dict('records')
    
    return train_files, balanced_val, class_weights




def get_transforms_specific(
    spatial_size: Tuple[int, int, int],
    stage: str = "train",
    class_name: str = None
) -> Compose:
    """
    Get task-specific data transforms for training or validation.
    """
    base_transforms = [
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
    ]
    
    if stage == "train" and class_name is not None:
        # Define intensity augmentations
        intensity_augs = [
            RandAdjustContrastd(keys=["img"], prob=0.3, gamma=(0.8, 1.2)),
            RandShiftIntensityd(keys=["img"], prob=0.3, offsets=0.1),
            RandScaleIntensityd(keys=["img"], prob=0.3, factors=0.15),
            RandBiasFieldd(keys=["img"], prob=0.2, degree=3, coeff_range=(0.0, 0.1)),
            RandGaussianNoised(keys=["img"], prob=0.2, std=0.05),
            RandGibbsNoised(keys=["img"], prob=0.15, alpha=(0.0, 1.0)),
            RandKSpaceSpikeNoised(keys=["img"], prob=0.15, intensity_range=(0.9, 1.1)),
        ]
        
        # Define spatial augmentations (conservative)
        spatial_augs = [
            RandFlipd(keys=["img"], prob=0.3, spatial_axis=[0, 1, 2]),
            RandRotated(
                keys=["img"], prob=0.4,
                range_x=0.1, range_y=0.1, range_z=0.05,  # Reduced rotation
                mode="bilinear", padding_mode="border"
            ),
            RandAffined(
                keys=["img"], prob=0.3,  # Reduced probability
                scale_range=(0.05, 0.05, 0.02),  # Much smaller scaling
                translate_range=(3, 3, 2),  # Smaller translation
                mode="bilinear", padding_mode="border"
            ),
        ]
        
        # Very conservative elastic deformation (only for specific tasks)
        mild_elastic = [
            Rand3DElasticd(
                keys=["img"], prob=0.1,  # Very low probability
                sigma_range=(8, 12),     # Smoother deformation
                magnitude_range=(0.5, 1.0),  # Much smaller magnitude
                mode="bilinear", padding_mode="border"
            ),
        ]
        
        # Task-specific augmentation selection
        if class_name == "Noise":
            # Only spatial - avoid noise augmentations
            aug_transforms = spatial_augs
            
        elif class_name == "Zipper":
            # Spatial + very mild intensity + rare elastic (zipper can have slight geometric component)
            safe_intensity = [
                RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.9, 1.1)),
                RandShiftIntensityd(keys=["img"], prob=0.2, offsets=0.05),
            ]
            aug_transforms = spatial_augs + safe_intensity + mild_elastic
            
        elif class_name == "Positioning":
            # Only intensity - NO spatial transforms that change positioning
            aug_transforms = intensity_augs
            
        elif class_name == "Banding":
            # Only spatial - avoid intensity changes
            aug_transforms = spatial_augs
            
        elif class_name == "Motion":
            # Only intensity - NO spatial transforms that simulate motion
            aug_transforms = intensity_augs
            
        elif class_name == "Contrast":
            # Only spatial - NO contrast/intensity modifications
            aug_transforms = spatial_augs
            
        elif class_name == "Distortion":
            # Only intensity - NO spatial distortions
            aug_transforms = intensity_augs
            
        else:
            # Fallback: conservative combination
            aug_transforms = intensity_augs + spatial_augs
        
        transforms = base_transforms + aug_transforms + [
            CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
            ToTensord(keys=["img"], dtype=torch.float32),
        ]
        
    else:
        transforms = base_transforms + [
            CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
            ToTensord(keys=["img"], dtype=torch.float32)
        ]
    
    return Compose(transforms)


def get_transforms(
    spatial_size: Tuple[int, int, int],
    stage: str = "train"
) -> Compose:
    """
    Get data transforms for training or validation.
    """
    base_transforms = [
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
    ]
    
    if stage == "train":
        transforms = base_transforms + [
            # Data augmentation for contrastive learning
            RandShiftIntensityd(keys=["img"], prob=0.3, offsets=0.1),
            RandScaleIntensityd(keys=["img"], prob=0.3, factors=0.1),
            RandRotated(
                keys=["img"], prob=0.4,
                range_x=0.1, range_y=0.1, range_z=0.1
            ),
            RandAffined(
                keys=["img"],
                prob=0.5,
                scale_range=(0.15, 0.15, 0.15),  # Less scaling in slice direction
                translate_range=(5, 5, 5),       # Less translation in slice direction
                mode="bilinear",
                padding_mode="border"
            ),
            CenterSpatialCropd(
                keys=["img"], roi_size=spatial_size
            ),
            SpatialPadd(
                keys=["img"], method="symmetric", spatial_size=spatial_size
            ),
            ToTensord(keys=["img"], dtype=torch.float32),
        ]
    else:
        # To ensure the same spatial size for all images
        transforms = base_transforms + [
            CenterSpatialCropd(
                keys=["img"], roi_size=spatial_size
            ),
            SpatialPadd(
                keys=["img"], method="symmetric", spatial_size=spatial_size
            ),
            ToTensord(keys=["img"], dtype=torch.float32)
        ]
    
    return Compose(transforms)


def train_model(
    class_name: str,
    train_files: List[Dict],
    val_files: List[Dict],
    config: dict,
    class_weights: torch.Tensor,
    load_model_path: Optional[str] = None
):
    """
    Train the ordinal model using Focal+EMD loss (no contrastive, no prototype).
    """
    os.makedirs(config['results_dir'], exist_ok=True)
    wandb.finish()

    # Add runtime parameters to config
    config = config.copy()
    config['class_name'] = class_name
    config['class_weights'] = (
        class_weights.cpu().tolist() if isinstance(class_weights, torch.Tensor) 
        else class_weights
    )
    config['load_model_path'] = load_model_path

    wandb_logger = WandbLogger(
        project=config['wandb_project'],
        name=f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_dir=config['results_dir'],
        config=config
    )
    try:
        wandb.config.update(config, allow_val_change=True)
    except Exception as e:
        print(f"Warning: wandb.config.update failed: {e}")

    # train_transforms = get_transforms_specific(config['spatial_size'], "train", class_name)
    # val_transforms = get_transforms_specific(config['spatial_size'], "val", class_name)

    train_transforms = get_transforms(config['spatial_size'], "train")
    val_transforms = get_transforms(config['spatial_size'], "val")
    train_dataset = Dataset(data=train_files, transform=train_transforms)
    val_dataset = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    if load_model_path is not None:
        print(f"Loading model from {load_model_path}")
        model = OrdinalFocalEMDLightningModule.load_from_checkpoint(
            load_model_path
        )
        trainer = None
    else:
        print("Training new model")
        model = OrdinalFocalEMDLightningModule(
            num_classes=config['model_params']['num_classes'],
            spatial_dims=config['model_params']['spatial_dims'],
            in_channels=config['model_params']['in_channels'],
            learning_rate=config['learning_rate'],
            weight_decay=config['model_params']['weight_decay'],
            max_epochs=config['num_epochs'],
            focal_gamma=config['model_params']['focal_gamma'],
            class_weights=class_weights,
            emd_weight=config['model_params']['emd_weight'],
            focal_weight=config['model_params']['focal_weight']
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(config['results_dir'], "checkpoints"),
            filename=f"{class_name}_{{epoch:02d}}_{{val/total_loss:.3f}}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            patience=30,
            mode="min",
            verbose=True
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            accelerator=config['trainer_params']['accelerator'],
            devices=config['trainer_params']['devices'],
            precision=config['trainer_params']['precision'],
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=wandb_logger,
            log_every_n_steps=config['trainer_params']['log_every_n_steps'],
            gradient_clip_val=config['trainer_params']['gradient_clip_val'],
            accumulate_grad_batches=config['trainer_params']['accumulate_grad_batches'],
            deterministic=config['trainer_params']['deterministic']
        )
        trainer.fit(model, train_loader, val_loader)
        best_model_path = checkpoint_callback.best_model_path
        model = OrdinalFocalEMDLightningModule.load_from_checkpoint(
            best_model_path
        )
        final_model_path = os.path.join(config['results_dir'],
                                        f"{class_name}_finalmodel.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"Saved final model to {final_model_path}")
    if trainer is not None:
        trainer.test(model, val_loader)

    # --- CONFUSION MATRIX LOGIC ---
    # Create evaluation dataloaders with smaller batch size for evaluation
    eval_train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False
    )
    eval_val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False
    )
    eval_dataloaders = {"train": eval_train_loader, "val": eval_val_loader}

    # Use the evaluate_model method from OrdinalFocalEMDLightningModule if available
    evaluation_results = None
    if hasattr(model, 'evaluate_model'):
        evaluation_results = model.evaluate_model(
            eval_dataloaders, 
            device=config['device'], 
            class_name=class_name, 
            wandb_logger=wandb_logger
        )
    else:
        evaluation_results = {}

    # Collect predictions for confusion matrix plotting
    collect_predictions_emd(model, eval_dataloaders, config['device'])
    
    class_names = [str(i) for i in range(3)]
    # Train split
    if hasattr(model, 'train_preds') and model.train_preds:
        y_pred_train = torch.cat(model.train_preds, dim=0).numpy()
        y_true_train = torch.cat(model.train_labels, dim=0).numpy()
        plot_confusion_matrix(
            y_true_train, y_pred_train, class_names, 'train',
            config['results_dir'], class_name, wandb_logger
        )
    # Validation split
    if hasattr(model, 'val_preds') and model.val_preds:
        y_pred_val = torch.cat(model.val_preds, dim=0).numpy()
        y_true_val = torch.cat(model.val_labels, dim=0).numpy()
        plot_confusion_matrix(
            y_true_val, y_pred_val, class_names, 'val',
            config['results_dir'], class_name, wandb_logger
        )
    # --- END CONFUSION MATRIX LOGIC ---

    return model, trainer, evaluation_results


def collect_predictions_emd(model, dataloaders, device):
    """Collect predictions for confusion matrix plotting."""
    model.eval()
    model = model.to(device)
    model.train_preds = []
    model.train_labels = []
    model.val_preds = []
    model.val_labels = []
    
    with torch.no_grad():
        if 'train' in dataloaders:
            for batch in dataloaders['train']:
                images = batch["img"].to(device)
                labels = batch["label"]
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                model.train_preds.append(preds.cpu())
                model.train_labels.append(labels.cpu())
        if 'val' in dataloaders:
            for batch in dataloaders['val']:
                images = batch["img"].to(device)
                labels = batch["label"]
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                model.val_preds.append(preds.cpu())
                model.val_labels.append(labels.cpu())


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train or evaluate ordinal contrastive model"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Path to results directory containing pre-trained models"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="If set, only run this task (e.g., 'Noise', 'Zipper', etc.)"
    )
    args = parser.parse_args()
    
    # Configuration
    config = {
        "bids_dir": "data/LISA2025/BIDS_norm",
        "qc_csv_path": "data/LISA2025/BIDS_norm/LISA_2025_bids.csv",
        "augmented_dir": "data/LISA2025/BIDS_norm/derivatives/augmented_FINAL_Distortion",
        "augmented_qc_csv_path": (
            "data/LISA2025/BIDS_norm/derivatives/augmented_FINAL_Distortion/augmented_labels.csv"
        ),
        "spatial_size": (150, 150, 150),
        "batch_size": 8,
        "num_epochs": 70,
        "learning_rate": 1e-4,
        "random_state": 42,
        "results_dir": "results_focal_augmented_onlyfocal_final",
        "num_workers": 2,
        "wandb_project": "lisa2025-ordinal-fullfocal-augmented",
        "model_params": {
            "num_classes": 3,
            "spatial_dims": 3,
            "in_channels": 1,
            "weight_decay": 1e-5,
            "focal_gamma": 2.0,
            "emd_weight": 0.0,
            "focal_weight": 1.0
        },
        "trainer_params": {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": '16-mixed',
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 2,
            "deterministic": False,
            "log_every_n_steps": 10
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_raw_only": False
    }
    
    # Task to train on
    all_tasks = [
        "Zipper", "Positioning", "Noise", "Banding",
        "Motion", "Contrast", "Distortion"
    ]

    # Determine which tasks to run
    if args.task:
        if args.task not in all_tasks:
            raise ValueError(f"Invalid task '{args.task}'. Must be one of: {all_tasks}")
        tasks = [args.task]
        single_task_mode = True
    else:
        tasks = all_tasks
        single_task_mode = False

    # Store all evaluation results
    all_evaluation_results = {}
    
    # Train model for each task
    for task in tasks:
        print(f"\n{'='*50}")
        if args.load_model:
            print(f"Evaluating model for task: {task}")
        else:
            print(f"Training model for task: {task}")
        print(f"{'='*50}\n")
        
        # Determine model path for loading
        load_model_path = None
        if args.load_model:
            # Construct the path to the specific task model
            task_model_path = os.path.join(
                args.load_model, f"{task}_finalmodel.ckpt"
            )
            if os.path.exists(task_model_path):
                load_model_path = task_model_path
                print(f"Found model for {task} at {load_model_path}")
            else:
                print(f"Warning: Model not found for {task} at "
                      f"{task_model_path}")
                print("Skipping this task...")
                continue
        
        # try:
        # Prepare data
        train_files, val_files, class_weights = prepare_data(
            class_name=task,
            bids_dir=config["bids_dir"],
            qc_csv_path=config["qc_csv_path"],
            augmented_dir=config["augmented_dir"],
            augmented_qc_csv_path=config["augmented_qc_csv_path"],
            random_state=config["random_state"],
            use_raw_only=config["use_raw_only"]
        )
        
        # Train or load model
        model, trainer, evaluation_results = train_model(
            class_name=task,
            train_files=train_files,
            val_files=val_files,
            config=config,
            class_weights=class_weights,
            load_model_path=load_model_path
        )
        
        # Store evaluation results
        all_evaluation_results[task] = evaluation_results
        
        # Helper to make results JSON serializable
        def to_serializable(obj):
            import numpy as np
            import torch
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            else:
                return obj
        
        # Save results to disk for this task
        import json
        results_save_path = os.path.join(
            config["results_dir"],
            f"{task}_results.json"
        )
        with open(results_save_path, "w") as f:
            json.dump(to_serializable(evaluation_results), f, indent=2)
        print(f"Saved evaluation results for {task} to {results_save_path}")
        
        if args.load_model:
            print(f"Successfully evaluated model for {task}")
        else:
            print(f"Successfully trained model for {task}")
        wandb.finish()

        # except Exception as e:
        #    print(f"Error training model for {task}: {str(e)}")
        #    continue
    
    # If single-task mode, do not compute overall scores
    if single_task_mode:
        print("\nSingle-task mode: skipping overall F1 score computation.")
        return
    
    # Compute overall F1 scores across all tasks
    print("\n" + "="*80)
    print("OVERALL RESULTS ACROSS ALL TASKS")
    print("="*80)
    
    # Collect all predictions and labels for overall metrics
    all_predictions = []
    all_labels = []
    
    for task, task_results in all_evaluation_results.items():
        for split_name, split_results in task_results.items():
            all_predictions.extend(split_results['predictions'])
            all_labels.extend(split_results['labels'])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute overall F1 scores
    from sklearn.metrics import f1_score
    
    f1_micro_overall = f1_score(all_labels, all_predictions, average='micro')
    f1_macro_overall = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted_overall = f1_score(
        all_labels, all_predictions, average='weighted'
    )
    
    print("\nOverall F1 Scores (across all tasks and splits):")
    print(f"  F1 Micro: {f1_micro_overall:.4f}")
    print(f"  F1 Macro: {f1_macro_overall:.4f}")
    print(f"  F1 Weighted: {f1_weighted_overall:.4f}")
    
    # Also compute per-task averages
    print("\nPer-task average F1 scores:")
    for task, task_results in all_evaluation_results.items():
        task_f1_scores = []
        for split_name, split_results in task_results.items():
            task_f1_scores.append(split_results['f1_micro'])
        avg_f1 = np.mean(task_f1_scores)
        print(f"  {task}: {avg_f1:.4f}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Run training
    main()