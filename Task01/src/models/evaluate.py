import os
import json
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import monai
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score
)
from monai.networks.nets import DenseNet264
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    SpatialPadd,
    ToTensord,
)
from train import train  # Reuse data loading logic


def evaluate_model(
    class_name: str,
    results_dir: str,
    n_classes: int = 3,
    x: int = 256,
    y: int = 256,
    z: int = 256,
    device: str = "cuda:0",
    random_state: int = 42
):
    """Evaluate a trained model on the validation set.
    
    Args:
        class_name (str): Name of the class/artefact to evaluate
        results_dir (str): Directory containing the trained model
        n_classes (int, optional): Number of classes. Defaults to 3.
        x (int, optional): Input image x dimension. Defaults to 256.
        y (int, optional): Input image y dimension. Defaults to 256.
        z (int, optional): Input image z dimension. Defaults to 256.
        device (str, optional): Device to evaluate on. Defaults to "cuda:0".
        random_state (int, optional): Random seed for reproducibility. 
            Defaults to 42.
    """
    # Create task-specific results directory
    task_results_dir = os.path.join(results_dir, class_name)
    os.makedirs(task_results_dir, exist_ok=True)
    
    # Load data using the same logic as training
    train_filenames, val_filenames, train_labels, val_labels = train(
        class_name=class_name,
        n_classes=n_classes,
        x=x,
        y=y,
        z=z,
        device=device,
        random_state=random_state,
        results_dir=task_results_dir,
        return_data_only=True  # Only return data, don't train
    )
    
    # Create validation dataset
    val_files = [
        {"img": str(img), "label": int(label)} 
        for img, label in zip(val_filenames, val_labels)
    ]
    
    # Define transforms
    val_transforms = Compose([
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
        ToTensord(keys=["img"], dtype=torch.float32),
    ])
    
    # Create dataset and dataloader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model
    model = DenseNet264(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_classes
    ).to(device)
    
    # Load best model
    model_path = os.path.join(
        task_results_dir,
        f"best_metric_model_LISA_LF_{class_name}.pth"
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate
    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["img"].to(device).float()
            val_labels = val_data["label"].to(device).long()
            
            outputs = model(val_images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_micro": precision_score(
            all_labels, all_preds, average="micro"
        ),
        "precision_macro": precision_score(
            all_labels, all_preds, average="macro"
        ),
        "precision_weighted": precision_score(
            all_labels, all_preds, average="weighted"
        ),
        "recall_micro": recall_score(all_labels, all_preds, average="micro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "recall_weighted": recall_score(
            all_labels, all_preds, average="weighted"
        ),
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save metrics
    with open(os.path.join(task_results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good", "Moderate", "Bad"],
        yticklabels=["Good", "Moderate", "Bad"]
    )
    plt.title(f"Confusion Matrix - {class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(task_results_dir, "confusion_matrix.png"))
    plt.close()
    
    return metrics, cm


def evaluate_all_tasks(results_dir: str = "results_dense264"):
    """Evaluate all trained models.
    
    Args:
        results_dir (str, optional): Directory containing trained models.
            Defaults to "results_dense264".
    """
    tasks = [
        "Noise",
        "Zipper",
        "Positioning",
        "Banding",
        "Motion",
        "Contrast",
        # "Distortion"
    ]
    
    all_metrics = {}
    all_preds = []
    all_labels = []
    
    for task in tasks:
        print(f"\nEvaluating {task}...")
        metrics, _ = evaluate_model(task, results_dir)
        all_metrics[task] = metrics
        
        # Load task-specific predictions and labels
        task_results_dir = os.path.join(results_dir, task)
        
        # Get predictions and labels from the model evaluation
        model_path = os.path.join(
            task_results_dir,
            f"best_metric_model_LISA_LF_{task}.pth"
        )
        
        # Load data using the same logic as training
        _, val_filenames, _, val_labels = train(
            class_name=task,
            n_classes=3,
            device="cuda:0",
            results_dir=task_results_dir,
            return_data_only=True
        )
        
        # Create validation dataset
        val_files = [
            {"img": str(img), "label": int(label)} 
            for img, label in zip(val_filenames, val_labels)
        ]
        
        # Define transforms
        val_transforms = Compose([
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
                spatial_size=(256, 256, 256)
            ),
            ToTensord(keys=["img"], dtype=torch.float32),
        ])
        
        # Create dataset and dataloader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize model
        model = DenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=3
        ).to("cuda:0")
        
        # Load best model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Get predictions
        task_preds = []
        task_labels = []
        
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["img"].to("cuda:0").float()
                val_labels = val_data["label"].to("cuda:0").long()
                
                outputs = model(val_images)
                preds = torch.argmax(outputs, dim=1)
                
                task_preds.extend(preds.cpu().numpy())
                task_labels.extend(val_labels.cpu().numpy())
        
        all_preds.extend(task_preds)
        all_labels.extend(task_labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate micro and macro averages
    micro_macro_metrics = {
        "micro": {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="micro"),
            "precision": precision_score(
                all_labels, all_preds, average="micro"
            ),
            "recall": recall_score(
                all_labels, all_preds, average="micro"
            ),
        },
        "macro": {
            "accuracy": balanced_accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro"),
            "precision": precision_score(
                all_labels, all_preds, average="macro"
            ),
            "recall": recall_score(
                all_labels, all_preds, average="macro"
            ),
        }
    }
    
    # Add micro and macro averages to all_metrics
    all_metrics["micro_macro_averages"] = micro_macro_metrics
    
    # Save combined metrics
    with open(os.path.join(results_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    # Print micro and macro averages
    print("\nMicro and Macro Averages across all tasks:")
    for avg_type in ["micro", "macro"]:
        print(f"\n{avg_type.capitalize()} averages:")
        for metric, value in micro_macro_metrics[avg_type].items():
            print(f"  {metric}: {value:.4f}")
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--task",
        type=str,
        help="Specific task to evaluate (optional)",
        choices=[
            "Noise", "Zipper", "Positioning", "Banding",
            "Motion", "Contrast", "Distortion"
        ]
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_dense264",
        help="Directory containing trained models"
    )
    
    args = parser.parse_args()
    
    if args.task:
        evaluate_model(args.task, args.results_dir)
    else:
        evaluate_all_tasks(args.results_dir) 