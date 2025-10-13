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
from sklearn.model_selection import StratifiedGroupKFold
import os

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


def train(
    config: dict
):
    """Train a DenseNet model for medical image classification.
    
    Args:
        config (dict): Configuration dictionary containing all necessary parameters.
    """
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)

    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load raw and augmented data
    qc_df = pd.read_csv(config['qc_csv_path'])
    qc_df["data_type"] = "raw"
    
    aug_qc_df = pd.read_csv(config['augmented_qc_csv_path'])
    aug_qc_df["data_type"] = "augmented"
    
    # Combine dataframes
    combined_df = pd.concat([qc_df, aug_qc_df], ignore_index=True)

    print(f"Number of rows in combined_df: {len(combined_df)}")
    # Remove rows with nan values in the column filename
    combined_df = combined_df[combined_df['filename'].notna()]
    
    print(f"Number of rows in combined_df after removing nan values: {len(combined_df)}")
    
    # Use only raw data if specified
    if config.get('use_raw_only', True):
        combined_df = combined_df[combined_df["data_type"] == "raw"]
        print(f"Number of rows after filtering for raw only: {len(combined_df)}")
    else:
        print(f"Using both raw and augmented data: {len(combined_df)}")
    
    # Get all task columns
    task_columns = [
        "Noise", "Zipper", "Positioning", "Banding", 
        "Motion", "Contrast", "Distortion"
    ]
    
    # Filter data for the specific task
    # Keep samples where target task has value (0,1,2) and other tasks are 0
    task_mask = (
        (combined_df[config['class_name']].notna()) & 
        (combined_df[config['class_name']].isin([0, 1, 2])) &
        (combined_df[task_columns].sum(axis=1) == combined_df[config['class_name']])
    )
    combined_df = combined_df[task_mask]
    
    print(
        f"Number of rows after filtering for task {config['class_name']}: "
        f"{len(combined_df)}"
    )
    
    # Extract filenames and labels
    filenames = combined_df['filename'].astype(str).values
    labels = combined_df[config['class_name']].values
    subject_ids = combined_df['subject_id'].values
    
    # Split into train and validation sets using StratifiedGroupKFold
    # We use n_splits=5 to get a 80/20 split (1/5 for validation)
    group_kfold = StratifiedGroupKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=config['random_state']
    )
    train_idx, val_idx = next(
        group_kfold.split(filenames, labels, subject_ids)
    )
    
    train_filenames = filenames[train_idx]
    val_filenames = filenames[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    
    # Print data distribution
    print(f"\nData distribution for task {config['class_name']}:")
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
    
    # If only returning data, return here
    if config['return_data_only']:
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
    
    device = torch.device(config['device'])

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
            CenterSpatialCropd(keys=["img"], roi_size=(config['x'], config['y'], config['z'])),
            SpatialPadd(
                keys=["img"], 
                method="symmetric", 
                spatial_size=(config['x'], config['y'], config['z'])
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
                spatial_size=(config['x'], config['y'], config['z'])
            ),
            ToTensord(keys=["img"], dtype=torch.float32),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=config['n_classes'])])

    # Create datasets and dataloaders
    # Remove unused train_ds since we create epoch-specific ones
    
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=2, 
        num_workers=0, 
        pin_memory=torch.cuda.is_available()
    )

    # Initialize model
    model = DenseNet264(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=config['n_classes']
    ).to(device)

    # Ensure model parameters are float32
    model = model.float()

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['n_epoch']
    )

    # Training loop
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(log_dir=os.path.join(config['results_dir'], "tensorboard"))

    epoch_loss_values = []
    metric_values = []
    
    # Early stopping variables
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['n_epoch']):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config['n_epoch']}")
        
        # Subsample label 0 cases for this epoch
        label_0_indices = [
            i for i, data in enumerate(train_files) if data["label"] == 0
        ]
        label_0_subset = np.random.choice(
            label_0_indices, 
            size=int(len(label_0_indices) * 0.1), 
            replace=False
        )
        label_1_2_indices = [
            i for i, data in enumerate(train_files) if data["label"] in [1, 2]
        ]
        selected_indices = np.concatenate([label_0_subset, label_1_2_indices])
        
        # Create new dataset with subsampled data
        epoch_train_files = [
            train_files[i] for i in selected_indices
        ]
        
        # Print label distribution for this epoch
        label_counts = {}
        for data in epoch_train_files:
            label = data["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        print("\nLabel distribution in this epoch:")
        for label in sorted(label_counts.keys()):
            print(f"Label {label}: {label_counts[label]} samples")
        
        epoch_train_ds = monai.data.Dataset(
            data=epoch_train_files, 
            transform=train_transforms
        )
        epoch_train_loader = DataLoader(
            epoch_train_ds, 
            batch_size=2, 
            shuffle=True, 
            num_workers=0,
            pin_memory=torch.cuda.is_available()
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

                y_onehot = [
                    post_label(i) 
                    for i in decollate_batch(y, detach=False)
                ]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                metric_result = metrics_func(y_pred_act, y_onehot)
                metric_values.append(metric_result.item())

                if metric_result >= best_metric:
                    best_metric = metric_result
                    best_metric_epoch = epoch + 1
                    best_model_state = model.state_dict().copy()
                    torch.save(
                        model.state_dict(), 
                        os.path.join(
                            config['results_dir'],
                            f"best_metric_model_LISA_LF_{config['class_name']}.pth"
                        )
                    )
                    print("saved new best metric model")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config['early_stopping_patience']:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"No improvement for {config['early_stopping_patience']} epochs")
                        break
                
                print(
                    f"current epoch: {epoch + 1} "
                    f"current F1-score: {metric_result:.4f} "
                    f"best F1-score: {best_metric:.4f} "
                    f"at epoch {best_metric_epoch} "
                    f"patience: {patience_counter}/"
                    f"{config['early_stopping_patience']}"
                )
                writer.add_scalar("val_auc", metric_result, epoch + 1)

        np.save(
            os.path.join(config['results_dir'], f"loss_tr_{config['class_name']}.npy"), 
            epoch_loss_values
        )
        np.save(
            os.path.join(config['results_dir'], f"val_mean_{config['class_name']}.npy"), 
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
    writer.close()
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
    # Central config
    base_config = {
        'n_classes': 3,
        'x': 256,
        'y': 256,
        'z': 256,
        'bids_dir': "data/LISA2025/BIDS",
        'qc_csv_path': "data/LISA2025/BIDS/LISA_2025_bids.csv",
        'augmented_dir': "data/LISA2025/BIDS/derivatives/augmented_v2",
        'augmented_qc_csv_path': "data/LISA2025/BIDS/derivatives/augmented_v2/augmented_labels.csv",
        'n_epoch': 20,
        'device': "cuda:0",
        'random_state': 42,
        'early_stopping_patience': 10,
        'min_delta': 0.001,
        'return_data_only': False,
        'use_raw_only': True
    }
    # Train a model for each task
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Training model for task: {task}")
        print(f"{'='*50}\n")
        # Create task-specific results directory
        task_results_dir = os.path.join("results_dense264", task)
        os.makedirs(task_results_dir, exist_ok=True)
        # Update config for this task
        config = base_config.copy()
        config['class_name'] = task
        config['results_dir'] = task_results_dir
        # Train model for this task
        best_metric, best_epoch = train(config)
        print(f"\nCompleted training for {task}")
        print(f"Best metric: {best_metric:.4f} at epoch {best_epoch}")