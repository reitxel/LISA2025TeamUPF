import logging
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.networks.nets import DenseNet264
from monai.data import decollate_batch, DataLoader
from evaluation.metrics import metrics_func
from sklearn.model_selection import train_test_split
import os
import wandb
from datetime import datetime

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)

from utils.ordinal import OrdinalDenseNet, QWKLoss, compute_ordinal_metrics, QWKFocalLoss



def train_ordinal(
    class_name: str,
    n_classes: int = 3,
    x: int = 150,
    y: int = 150,
    z: int = 150,
    bids_dir: str = "data/LISA2025/BIDS",
    qc_csv_path: str = "data/LISA2025/BIDS/LISA_2025_bids.csv",
    augmented_dir: str = "data/LISA2025/BIDS/derivatives/augmented_v2",
    augmented_qc_csv_path: str = (
        "data/LISA2025/BIDS/derivatives/augmented_v2/augmented_labels.csv"
    ),
    n_epoch: int = 20,
    device: str = "cuda:0",
    random_state: int = 42,
    results_dir: str = "results_ordinal",
    early_stopping_patience: int = 10,
    min_delta: float = 0.001,
    return_data_only: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    qwk_weight: float = 0.7,
    focal_weight: float = 0.3,
    batch_size: int = 6,
):
    """Train an ordinal DenseNet model for medical image classification.
    
    Args:
        class_name (str): Name of the class/artefact to train on
        n_classes (int, optional): Number of classes. Defaults to 3.
        x (int, optional): Input image x dimension. Defaults to 256.
        y (int, optional): Input image y dimension. Defaults to 256.
        z (int, optional): Input image z dimension. Defaults to 256.
        bids_dir (str, optional): BIDS directory containing the data. 
        qc_csv_path (str, optional): Path to QC CSV file.
        augmented_dir (str, optional): Directory containing augmented data.
        augmented_qc_csv_path (str, optional): Path to augmented QC CSV file.
        n_epoch (int, optional): Number of training epochs. Defaults to 20.
        device (str, optional): Device to train on. Defaults to "cuda:0".
        random_state (int, optional): Random seed for reproducibility. 
            Defaults to 42.
        results_dir (str, optional): Directory to save results. 
            Defaults to "results_ordinal".
        early_stopping_patience (int, optional): Patience for early stopping. 
            Defaults to 10.
        min_delta (float, optional): Minimum change for early stopping. 
            Defaults to 0.001.
        return_data_only (bool, optional): If True, only return the data splits.
            Defaults to False.
        use_focal_loss (bool, optional): If True, use focal loss. 
            Defaults to True.
        focal_gamma (float, optional): Gamma for focal loss. Defaults to 2.0.
        qwk_weight (float, optional): Weight for QWK loss. Defaults to 0.7.
        focal_weight (float, optional): Weight for focal loss. Defaults to 0.3.
    """
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Initialize wandb
    wandb.init(
        project="lisa2025-qc-ordinal",
        name=f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "class_name": class_name,
            "n_classes": n_classes,
            "input_size": (x, y, z),
            "n_epoch": n_epoch,
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
            "random_state": random_state,
            "device": device,
        }
    )

    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load raw and augmented data
    qc_df = pd.read_csv(qc_csv_path)
    qc_df["data_type"] = "raw"
    
    aug_qc_df = pd.read_csv(augmented_qc_csv_path)
    aug_qc_df["data_type"] = "augmented"
    
    # Combine dataframes
    combined_df = pd.concat([qc_df, aug_qc_df], ignore_index=True)

    print(f"Number of rows in combined_df: {len(combined_df)}")
    # Remove rows with nan values in the column filename
    combined_df = combined_df[combined_df['filename'].notna()]
    
    print(f"Number of rows in combined_df after removing nan values: {len(combined_df)}")
    
    # Get all task columns
    task_columns = [
        "Noise", "Zipper", "Positioning", "Banding", 
        "Motion", "Contrast", "Distortion"
    ]
    
    # Filter data for the specific task
    # Keep samples where target task has value (0,1,2) and other tasks are 0
    task_mask = (
        (combined_df[class_name].notna()) & 
        (combined_df[class_name].isin([0, 1, 2])) &
        (combined_df[task_columns].sum(axis=1) == combined_df[class_name])
    )
    combined_df = combined_df[task_mask]
    
    print(
        f"Number of rows after filtering for task {class_name}: "
        f"{len(combined_df)}"
    )
    
    # Extract filenames and labels
    filenames = combined_df['filename'].astype(str).values
    labels = combined_df[class_name].values
    
    # Calculate class weights based on the full dataset (before sampling)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"\nOriginal class distribution for {class_name}:")
    for label, count in zip(unique_labels, label_counts):
        print(f"Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Calculate alpha weights (inverse frequency normalized)
    class_weights = len(labels) / (n_classes * label_counts)
    class_weights = class_weights / class_weights.mean()  # Normalize to mean=1
    
    print(f"\nClass weights (alpha): {class_weights}")
    
    # Log to wandb
    wandb.config.update({
        "use_focal_loss": use_focal_loss,
        "focal_gamma": focal_gamma,
        "qwk_weight": qwk_weight,
        "focal_weight": focal_weight,
        "class_weights": class_weights.tolist()
    })
    
    # Split into train and validation sets
    train_filenames, val_filenames, train_labels, val_labels = (
        train_test_split(
            filenames,
            labels,
            test_size=0.2, 
            random_state=random_state, 
            stratify=labels
        )
    )
    
    # Print data distribution
    print(f"\nData distribution for task {class_name}:")
    print(f"Total samples: {len(filenames)}")
    print(f"Training samples: {len(train_filenames)}")
    print(f"Validation samples: {len(val_filenames)}")
    print("\nLabel distribution in training set:")
    for label in sorted(np.unique(train_labels)):
        count = np.sum(train_labels == label)
        print(f"Label {label}: {count} samples")
    print("\nLabel distribution in validation set:")
    for label in sorted(np.unique(val_labels)):
        count = np.sum(val_labels == label)
        print(f"Label {label}: {count} samples")
    
    # Log data distribution to wandb
    wandb.log({
        "data/total_samples": len(filenames),
        "data/train_samples": len(train_filenames),
        "data/val_samples": len(val_filenames),
        "data/label_distribution": wandb.Table(
            data=[[label, np.sum(train_labels == label), np.sum(val_labels == label)]
                  for label in sorted(np.unique(train_labels))],
            columns=["label", "train_count", "val_count"]
        )
    })
    
    # If only returning data, return here
    if return_data_only:
        return train_filenames, val_filenames, train_labels, val_labels
    
    # Create train and validation data lists
    train_files = [
        {"img": str(img), "label": int(label)} 
        for img, label in zip(train_filenames, train_labels)
    ]
    val_files = [
        {"img": str(img), "label": int(label)} 
        for img, label in zip(val_filenames, val_labels)
    ]
    
    device = torch.device(device)

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"], 
                nonzero=False, 
                channel_wise=True
            ),
            CenterSpatialCropd(keys=["img"], roi_size=(x, y, z)),
            SpatialPadd(
                keys=["img"], 
                method="symmetric", 
                spatial_size=(x, y, z)
            ),
            ToTensord(keys=["img"], dtype=torch.float32),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"], 
                nonzero=False, 
                channel_wise=True
            ),
            SpatialPadd(
                keys=["img"], 
                method="symmetric", 
                spatial_size=(x, y, z)
            ),
            ToTensord(keys=["img"], dtype=torch.float32),
        ]
    )

    # Create datasets and dataloaders
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        num_workers=0, 
        pin_memory=torch.cuda.is_available()
    )

    # Initialize ordinal model
    model = OrdinalDenseNet(
        num_classes=n_classes,
        spatial_dims=3,
        in_channels=1
    ).to(device)

    # Ensure model parameters are float32
    model = model.float()

    # Use QWK loss for ordinal classification
    if use_focal_loss:
        loss_function = QWKFocalLoss(
            num_classes=n_classes,
            gamma=focal_gamma,
            alpha=class_weights,
            qwk_weight=qwk_weight,
            focal_weight=focal_weight
        )
        print(f"\nUsing QWKFocalLoss with gamma={focal_gamma}, "
              f"qwk_weight={qwk_weight}, focal_weight={focal_weight}")
    else:
        loss_function = QWKLoss(num_classes=n_classes, alpha=class_weights)
        print("\nUsing standard QWKLoss with class weights")

    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=n_epoch
    )

    # Training loop
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard"))

    epoch_loss_values = []
    metric_values = []
    
    # Early stopping variables
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epoch}")
        
        # Use all training data with class weights
        epoch_train_files = train_files
        
        # Print label distribution for this epoch
        label_counts = {}
        for data in epoch_train_files:
            label = data["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        print("\nLabel distribution in this epoch:")
        for label in sorted(label_counts.keys()):
            print(f"Label {label}: {label_counts[label]} samples")
        
        # Log epoch label distribution to wandb
        wandb.log({
            "epoch/label_distribution": wandb.Table(
                data=[[label, count] for label, count in label_counts.items()],
                columns=["label", "count"]
            ),
            "epoch": epoch
        })
        
        epoch_train_ds = monai.data.Dataset(
            data=epoch_train_files, 
            transform=train_transforms
        )
        epoch_train_loader = DataLoader(
            epoch_train_ds, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop the last incomplete batch
        )
        
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in epoch_train_loader:
            step += 1
            inputs = batch_data["img"].to(device).float()
            labels = batch_data["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(epoch_train_ds) // epoch_train_loader.batch_size
            if step % 50 == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                # Log batch metrics to wandb
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": epoch_len * epoch + step
                })
            writer.add_scalar(
                "train_loss", 
                loss.item(), 
                epoch_len * epoch + step
            )

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images = val_data["img"].to(device).float()
                    val_labels = val_data["label"].to(device).long()
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                # Compute ordinal metrics
                metrics = compute_ordinal_metrics(y, y_pred)
                metric_result = metrics['qwk']  # Use QWK as main metric
                metric_values.append(metric_result)

                # Log validation metrics to wandb
                wandb.log({
                    "val/qwk": metrics['qwk'],
                    "val/accuracy": metrics['accuracy'],
                    "val/accuracy_off_by_1": metrics['accuracy_off_by_1'],
                    "val/mae": metrics['mae'],
                    "epoch": epoch
                })

                if metric_result >= best_metric:
                    best_metric = metric_result
                    best_metric_epoch = epoch + 1
                    best_model_state = model.state_dict().copy()
                    torch.save(
                        model.state_dict(), 
                        os.path.join(
                            results_dir,
                            f"best_metric_model_LISA_LF_{class_name}.pth"
                        )
                    )
                    print("saved new best metric model")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"No improvement for {early_stopping_patience} epochs")
                        break
                
                print(
                    f"current epoch: {epoch + 1} "
                    f"current QWK: {metric_result:.4f} "
                    f"best QWK: {best_metric:.4f} "
                    f"at epoch {best_metric_epoch} "
                    f"patience: {patience_counter}/"
                    f"{early_stopping_patience}"
                )
                writer.add_scalar("val_qwk", metric_result, epoch + 1)

        # Log epoch metrics to wandb
        wandb.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch": epoch
        })

        np.save(
            os.path.join(results_dir, f"loss_tr_{class_name}.npy"), 
            epoch_loss_values
        )
        np.save(
            os.path.join(results_dir, f"val_mean_{class_name}.npy"), 
            metric_values
        )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model from epoch", best_metric_epoch)

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )
    
    # Log final results to wandb
    wandb.log({
        "final/best_qwk": best_metric,
        "final/best_epoch": best_metric_epoch
    })
    
    # Save model to wandb
    wandb.save(os.path.join(results_dir, f"best_metric_model_LISA_LF_{class_name}.pth"))
    
    writer.close()
    wandb.finish()
    return best_metric, best_metric_epoch


if __name__ == "__main__":
    # List of all tasks/classes to train
    tasks = [
        "Noise",
        "Zipper",
        "Positioning",
        "Banding",
        "Motion",
        "Contrast",
        "Distortion"
    ]
    
    # Train a model for each task
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Training model for task: {task}")
        print(f"{'='*50}\n")
        
        # Create task-specific results directory
        task_results_dir = os.path.join("results_ordinal", task)
        os.makedirs(task_results_dir, exist_ok=True)
        
        # Train model for this task
        best_metric, best_epoch = train_ordinal(
            class_name=task,
            results_dir=task_results_dir
        )
        
        print(f"\nCompleted training for {task}")
        print(f"Best metric: {best_metric:.4f} at epoch {best_epoch}")