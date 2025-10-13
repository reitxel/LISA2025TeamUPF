"""
Ensemble Evaluation Script

This script performs model ensembling for predictions on both the validation split and a separate validation set.
It loads multiple trained models, performs predictions for each, ensembles the results, and outputs:
- CSV with final results for the separate validation set
- Results for the validation split
- A table comparing each individual model with the ensemble (metrics for "All" tasks)

No main function is used; all code is modular and ready for import or direct execution.
"""

# =========================
# 1. Imports
# =========================
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
import glob
import random
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import mode
import pickle
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    CenterSpatialCropd, SpatialPadd, ToTensord
)
import joblib
import nibabel as nib
import json
import sys

# HACK but we are past the point of caring
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classical.feat_extraction import NeonatalMRIQualityAssessment


# =========================
# 2. Parameters and Paths
# =========================

# List of parent result folders containing all 7 task models (edit as needed)
MODEL_FOLDERS = [
    {"folder": "results_classical_FINAL", "model_type": "classical"},
    {"folder": "baseline/results_aug_online", "model_type": "densenet"},
    {"folder": "results_emd_augmented_onlyemd_final", "model_type": "focalemd"},
    {"folder": "results_focal_augmented_onlyfocal_final", "model_type": "focalemd"},
    {"folder": "results_emd_augmented_both_final", "model_type": "focalemd"},
    # Add more as needed
]
# NEW: Specify which model indexes to use for the ensemble
USE_FOR_ENSEMBLE = [1,2,3,4]  # <-- Edit this list to select model indexes for the ensemble

# Data locations
SEPARATE_VALIDATION_DIR = "data/LISA2025/Task_1_Validation/"

# Random seed
RANDOM_STATE = 42

# Use Bayesian network for probability correction?
USE_BAYESIAN_NETWORK = True
# Path to CSV for BN training (should have all task columns)
BAYESIAN_TRAIN_CSV = "data/LISA2025/BIDS_norm/LISA_2025_bids.csv"

# Confidence threshold for hard evidence in BN
BAYESIAN_CONFIDENCE_THRESHOLD = 0.7
# Mixing parameter alpha for BN adjustment
BAYESIAN_ALPHA = 0.2
# Epsilon for numerical stability
BAYESIAN_EPSILON = 1e-8
# Optional: path to save/load BN model
BAYESIAN_MODEL_PATH = None

# Output paths
OUTPUT_DIR = "ensemble_results_FINAL_version"

# Binary collapse toggle
BINARY_COLLAPSE = False  # Set to True to enable binary collapse strategy

# create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "LISA_LF_QC_predictions.csv")
OUTPUT_COMPARISON_TABLE_PATH = os.path.join(OUTPUT_DIR, "ensemble_comparison_table.csv")

# Metrics CSV prefix (user can set this)
METRICS_CSV_PREFIX = "model_"  # <-- Set your desired prefix here

# Model type
MODEL_TYPE = "focalemd"
# Device
DEVICE = "cuda"
# Spatial size for inference
SPATIAL_SIZE = (150, 150, 150)

# Inference batch size (user can set this)
INFERENCE_BATCH_SIZE = 32  # <-- Set your desired batch size here

# =========================
# 3. Model Loading
# =========================

def load_model(model_folder, use_bayesian=False, model_type="focalemd", device="cuda"):
    """
    Loads all 7 task models from the specified folder.
    If use_bayesian is True, loads the Bayesian version (handled later).
    Returns a dict: {task: model}
    """
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    models = {}
    if model_type == "densenet":
        import torch
        import monai
        n_classes = 3
        for task in tasks:
            model_path = os.path.join(model_folder, f"best_metric_model_LISA_LF_{task}.pth")
            if os.path.exists(model_path):
                try:
                    model = monai.networks.nets.DenseNet264(
                        spatial_dims=3, in_channels=1, out_channels=n_classes
                    )
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint)
                    model = model.to(device)
                    model.eval()
                    models[task] = model
                except Exception as e:
                    print(f"Error loading DenseNet model for {task} in {model_folder}: {e}")
            else:
                print(f"DenseNet model checkpoint not found for {task} in {model_folder}")
        return models
    else:
        from model import OrdinalContrastiveLightningModule, OrdinalFocalEMDLightningModule
        if model_type == "contrastive":
            model_class = OrdinalContrastiveLightningModule
        else:
            model_class = OrdinalFocalEMDLightningModule
        for task in tasks:
            model_path = os.path.join(model_folder, f"{task}_finalmodel.ckpt")
            if os.path.exists(model_path):
                try:
                    model = model_class.load_from_checkpoint(model_path)
                    model = model.to(device)
                    model.eval()
                    models[task] = model
                except Exception as e:
                    print(f"Error loading model for {task} in {model_folder}: {e}")
            else:
                print(f"Model checkpoint not found for {task} in {model_folder}")
        return models

# =========================
# 4. Data Loading
# =========================

def set_seed(seed):
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def create_per_task_validation_splits(raw_csv_path, random_state=42):
    """
    Create 7 separate validation splits, one for each task, using the same logic as train_emd.py.
    Returns: dict {task: DataFrame of validation split for that task}
    """
    # This function mimics the validation split logic from training, ensuring fair evaluation per task
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    raw_df = pd.read_csv(raw_csv_path)
    raw_df = raw_df[raw_df['filename'].notna()].copy()
    raw_df['data_type'] = 'raw'
    val_splits = {}
    for task in tasks:
        set_seed(random_state)
        # Single-task filter: at most one other task labeled 1/2
        other_tasks = [col for col in tasks if col != task]
        other_labeled_count = (raw_df[other_tasks].isin([1, 2])).sum(axis=1)
        other_task_mask = other_labeled_count <= 1
        task_df = raw_df[other_task_mask].copy()
        # Only use raw data
        task_df = task_df[task_df['data_type'] == 'raw']
        # Only use rows with valid label for this task
        mask = task_df[task].isin([0, 1, 2])
        subdf = task_df[mask]
        if subdf.empty:
            val_splits[task] = pd.DataFrame()
            continue
        filenames = subdf['filename'].astype(str).values
        labels = subdf[task].astype(int).values
        subjects = [os.path.basename(f).split('_')[0] for f in filenames]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        try:
            train_idx, val_idx = next(sgkf.split(filenames, labels, subjects))
            val_subjects = set(np.array(subjects)[val_idx])
        except Exception as e:
            print(f"Error in validation split for task {task}: {e}")
            val_splits[task] = pd.DataFrame()
            continue
        # Now filter the original df to only those validation subjects
        raw_df['subject'] = [os.path.basename(f).split('_')[0] for f in raw_df['filename'].astype(str).values]
        val_split_df = raw_df[raw_df['subject'].isin(val_subjects)].copy()
        val_splits[task] = val_split_df
    return val_splits

# =========================
# 5. Prediction and Ensembling
# =========================

def ensemble_predictions(predictions_list, method="average"):
    """
    Ensembles predictions from multiple models.
    predictions_list: list of numpy arrays (N, num_classes)
    method: "average" (default) or "majority"
    Returns: ensembled predictions (N, num_classes)
    """
    if method == "average":
        return np.mean(predictions_list, axis=0)
    elif method == "majority":
        # For classification: majority vote on argmax
        preds = np.array([np.argmax(p, axis=1) for p in predictions_list])
        from scipy.stats import mode
        return mode(preds, axis=0)[0][0]
    else:
        raise ValueError("Unknown ensembling method")

# =========================
# 6. Metrics Calculation
# =========================

def compute_metrics(y_true, y_pred, average="macro"):
    """
    Computes general metrics for all tasks.
    Returns: dict of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
    }
    return metrics

def compute_all_metrics(pred_df, gt_df, task=None):
    """
    Computes extended metrics for all tasks concatenated ("All" column) for a given predictions DataFrame and ground truth DataFrame.
    If 'task' is provided, computes metrics only for that task.
    Returns a dict of metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
    def calc_metrics(y_true, y_pred):
        # Compute per-class accuracies for macro/weighted accuracy
        from collections import Counter
        import numpy as np
        classes = np.unique(y_true + y_pred) if y_true and y_pred else []
        class_accuracies = {}
        supports = {}
        for c in classes:
            mask = [yt == c for yt in y_true]
            if sum(mask) > 0:
                class_acc = np.mean([y_pred[i] == c for i, m in enumerate(mask) if m])
                class_accuracies[c] = class_acc
                supports[c] = sum(mask)
            else:
                class_accuracies[c] = np.nan
                supports[c] = 0
        macro_acc = np.nanmean(list(class_accuracies.values())) if class_accuracies else None
        total = sum(supports.values())
        weighted_acc = (
            np.nansum([class_accuracies[c] * supports[c] for c in classes]) / total if total > 0 else None
        )
        micro_acc = accuracy_score(y_true, y_pred) if y_true else None
        # Composite metric: mean of weighted accuracy, f1, f2, precision, recall
        weighted_vals = [weighted_acc,
                        f1_score(y_true, y_pred, average="weighted") if y_true else None,
                        fbeta_score(y_true, y_pred, beta=2, average="weighted") if y_true else None,
                        precision_score(y_true, y_pred, average="weighted") if y_true else None,
                        recall_score(y_true, y_pred, average="weighted") if y_true else None]
        composite = np.nanmean([v for v in weighted_vals if v is not None]) if any(v is not None for v in weighted_vals) else None
        result = {
            "accuracy": micro_acc,
            "accuracy_micro": micro_acc,
            "accuracy_macro": macro_acc,
            "accuracy_weighted": weighted_acc,
            "f1_micro": f1_score(y_true, y_pred, average="micro") if y_true else None,
            "f1_macro": f1_score(y_true, y_pred, average="macro") if y_true else None,
            "f1_weighted": f1_score(y_true, y_pred, average="weighted") if y_true else None,
            "f2_micro": fbeta_score(y_true, y_pred, beta=2, average="micro") if y_true else None,
            "f2_macro": fbeta_score(y_true, y_pred, beta=2, average="macro") if y_true else None,
            "f2_weighted": fbeta_score(y_true, y_pred, beta=2, average="weighted") if y_true else None,
            "precision_micro": precision_score(y_true, y_pred, average="micro") if y_true else None,
            "precision_macro": precision_score(y_true, y_pred, average="macro") if y_true else None,
            "precision_weighted": precision_score(y_true, y_pred, average="weighted") if y_true else None,
            "recall_micro": recall_score(y_true, y_pred, average="micro") if y_true else None,
            "recall_macro": recall_score(y_true, y_pred, average="macro") if y_true else None,
            "recall_weighted": recall_score(y_true, y_pred, average="weighted") if y_true else None,
            "Composite": composite,
        }
        return result
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    if task == "All":
        y_true = gt_df["All"].astype(int).tolist()
        y_pred = pred_df["All"].astype(int).tolist()
        return calc_metrics(y_true, y_pred)
    if task is not None:
        if task not in pred_df.columns or task not in gt_df.columns:
            return {k: None for k in ["accuracy", "f1_micro", "f1_macro", "f1_weighted", "f2_micro", "f2_macro", "f2_weighted", "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted"]}
        mask = (~pd.isna(pred_df[task])) & (~pd.isna(gt_df[task]))
        y_true = gt_df.loc[mask, task].astype(int).tolist()
        y_pred = pred_df.loc[mask, task].astype(int).tolist()
        return calc_metrics(y_true, y_pred)
    # Default: all tasks
    y_true = []
    y_pred = []
    for t in tasks:
        mask = (~pd.isna(pred_df[t])) & (~pd.isna(gt_df[t]))
        y_true.extend(gt_df.loc[mask, t].astype(int).tolist())
        y_pred.extend(pred_df.loc[mask, t].astype(int).tolist())
    return calc_metrics(y_true, y_pred)

def build_comparison_table(model_preds_list, ensemble_preds, gt_df, model_names=None):
    """
    Builds a comparison table for all models and the ensemble.
    model_preds_list: list of DataFrames (one per model)
    ensemble_preds: DataFrame of ensemble predictions
    gt_df: ground truth DataFrame
    model_names: optional list of model names
    Returns: DataFrame with metrics for each model and the ensemble (columns)
    """
    all_tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_preds_list))]
    metrics_dict = {}
    for name, preds in zip(model_names, model_preds_list):
        # Ensure all task columns exist
        for task in all_tasks:
            if task not in preds.columns:
                preds[task] = np.nan
            if task not in gt_df.columns:
                gt_df[task] = np.nan
        metrics_dict[name] = compute_all_metrics(preds, gt_df)
    # Also for ensemble
    for task in all_tasks:
        if task not in ensemble_preds.columns:
            ensemble_preds[task] = np.nan
        if task not in gt_df.columns:
            gt_df[task] = np.nan
    metrics_dict["Ensemble"] = compute_all_metrics(ensemble_preds, gt_df)
    return pd.DataFrame(metrics_dict)

# =========================
# 8. Main Evaluation Logic (modular, not in __main__)
# =========================

def run_ensemble_evaluation():
    """
    Runs the full ensemble evaluation pipeline.
    """
    # This function orchestrates the entire ensemble evaluation process
    device = DEVICE
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    # 1. Create per-task validation splits from raw CSV (for metrics only)
    print("Creating per-task validation splits...")
    per_task_val_splits = create_per_task_validation_splits(BAYESIAN_TRAIN_CSV, random_state=RANDOM_STATE)
    # 2. Prepare transforms
    transforms = get_inference_transforms()
    # 3. Sequentially load models, predict, and release memory
    print("Processing ensemble models sequentially to minimize GPU memory usage...")
    model_preds_list_val = []
    model_probs_list_val = []  # NEW: store per-model per-task per-sample probabilities
    per_model_metrics_tables = []  # Store per-model metrics tables for later
    model_names = []
    for idx, folder_info in enumerate(MODEL_FOLDERS):
        folder = folder_info["folder"]
        model_type = folder_info.get("model_type", "focalemd")
        print(f"Loading models from folder {idx+1}/{len(MODEL_FOLDERS)}: {folder} (type: {model_type})")
        if model_type == "classical":
            mask_dir = "data/LISA2025/Task_1_Validation_masks/"  # Adjust as needed
            classical_models = load_classical_models(folder, tasks)
            preds_val = {}
            metrics_per_task = {}
            probs_val = {task: [] for task in tasks}  # NEW: store probabilities
            filenames_val = None
            for task in tasks:
                val_split = per_task_val_splits[task]
                if val_split.empty:
                    preds_val[task] = pd.DataFrame()
                    probs_val[task] = []
                    continue
                print(f"  Evaluating on validation split for task '{task}' ({len(val_split)} samples)...")
                # Use new proba function
                proba_dict, filenames = batch_predict_classical_proba(classical_models, val_split["filename"].tolist(), mask_dir, folder, [task])
                # proba_dict[task] is a list of np.array([p0, p1, p2])
                probs_val[task] = proba_dict[task]
                filenames_val = filenames  # Should match val_split["filename"].tolist()
                # For metrics, get hard predictions
                hard_preds = [int(binary_collapse_predict(p)) if BINARY_COLLAPSE and not np.isnan(p).all() else (int(np.nanargmax(p)) if not np.isnan(p).all() else None) for p in proba_dict[task]]
                df_pred = pd.DataFrame({"filename": filenames, task: hard_preds})
                preds_val[task] = df_pred
                metrics_per_task[task] = compute_all_metrics(df_pred, val_split, task=task)
            print(f"Finished evaluating classical model {idx+1}/{len(MODEL_FOLDERS)} on validation splits.")
            model_preds_list_val.append(preds_val)
            model_probs_list_val.append(probs_val)
            model_names.append(os.path.basename(folder.rstrip('/')))
            metrics_df = pd.DataFrame(metrics_per_task).T
            metrics_df.index.name = 'tasks'
            metrics_df = metrics_df.transpose()
            metrics_df.insert(0, 'metrics', metrics_df.index)
            metrics_df = metrics_df.reset_index(drop=True)
            folder_name = os.path.basename(folder.rstrip('/'))
            metrics_csv_path = os.path.join(OUTPUT_DIR, f"{METRICS_CSV_PREFIX}{folder_name}_metrics.csv")
            save_csv_and_latex(metrics_df, metrics_csv_path, latex_caption=f"Metrics for {folder_name} on validation splits.")
            print(f"Saved metrics CSV and LaTeX for {folder_name} to {metrics_csv_path}")
            per_model_metrics_tables.append(metrics_df)
            del classical_models
            import torch
            torch.cuda.empty_cache()
            print(f"Released classical models from {folder} and cleared GPU memory.")
        else:
            model_dict = load_model(folder, use_bayesian=USE_BAYESIAN_NETWORK, model_type=model_type, device=device)
            print(f"Evaluating model {idx+1}/{len(MODEL_FOLDERS)} on validation splits...")
            preds_val = {}
            metrics_per_task = {}
            probs_val = {task: [] for task in tasks}  # NEW: store probabilities
            filenames_val = None
            for task in tasks:
                val_split = per_task_val_splits[task]
                if val_split.empty:
                    preds_val[task] = pd.DataFrame()
                    probs_val[task] = []
                    continue
                print(f"  Evaluating on validation split for task '{task}' ({len(val_split)} samples)...")
                # Predict probabilities for all samples in this split
                image_paths = val_split["filename"].tolist()
                batch_size = INFERENCE_BATCH_SIZE
                all_probs = []
                for batch_start in range(0, len(image_paths), batch_size):
                    batch_end = min(batch_start + batch_size, len(image_paths))
                    batch_paths = image_paths[batch_start:batch_end]
                    batch_probs = predict_image_probabilities_batch({task: model_dict[task]}, batch_paths, transforms, device)
                    all_probs.extend(batch_probs[task])
                probs_val[task] = all_probs
                filenames_val = image_paths
                # For metrics, get hard predictions
                hard_preds = [int(binary_collapse_predict(p)) if BINARY_COLLAPSE and not np.isnan(p).all() else (int(np.nanargmax(p)) if not np.isnan(p).all() else None) for p in all_probs]
                df_pred = pd.DataFrame({"filename": image_paths, task: hard_preds})
                preds_val[task] = df_pred
                metrics_per_task[task] = compute_all_metrics(df_pred, val_split, task=task)
            print(f"Finished evaluating model {idx+1}/{len(MODEL_FOLDERS)} on validation splits.")
            model_preds_list_val.append(preds_val)
            model_probs_list_val.append(probs_val)
            model_names.append(os.path.basename(folder.rstrip('/')))
            metrics_df = pd.DataFrame(metrics_per_task).T
            metrics_df.index.name = 'tasks'
            metrics_df = metrics_df.transpose()
            metrics_df.insert(0, 'metrics', metrics_df.index)
            metrics_df = metrics_df.reset_index(drop=True)
            folder_name = os.path.basename(folder.rstrip('/'))
            metrics_csv_path = os.path.join(OUTPUT_DIR, f"{METRICS_CSV_PREFIX}{folder_name}_metrics.csv")
            save_csv_and_latex(metrics_df, metrics_csv_path, latex_caption=f"Metrics for {folder_name} on validation splits.")
            print(f"Saved metrics CSV and LaTeX for {folder_name} to {metrics_csv_path}")
            per_model_metrics_tables.append(metrics_df)
            del model_dict
            import torch
            torch.cuda.empty_cache()
            print(f"Released models from {folder} and cleared GPU memory.")
    # 4. Optionally train/load Bayesian Network
    bn = None
    bn_params = dict(epsilon=BAYESIAN_EPSILON, confidence_threshold=BAYESIAN_CONFIDENCE_THRESHOLD, alpha=BAYESIAN_ALPHA)
    if USE_BAYESIAN_NETWORK:
        print("Training or loading Bayesian Network...")
        bn = train_or_load_bayesian_network(BAYESIAN_TRAIN_CSV, tasks, model_path=BAYESIAN_MODEL_PATH)
        print("Bayesian Network ready.")
    # 5. Ensemble predictions (optionally with BN) on validation splits
    print("Running ensemble predictions on validation splits...")
    # Use only selected models for ensemble
    selected_model_probs_list_val = [model_probs_list_val[i] for i in USE_FOR_ENSEMBLE]
    ensemble_preds_val = ensemble_predict_from_probabilities(
        selected_model_probs_list_val, tasks, per_task_val_splits["Noise"]["filename"].tolist(),
        use_bn=USE_BAYESIAN_NETWORK, bn=bn, bn_params=bn_params, binary_collapse=BINARY_COLLAPSE
    )
    print("Finished ensemble predictions on validation splits.")

    # Build a single comparison table for all models and the ensemble (metrics for 'All' tasks)
    # Each column: model/folder name, 'Ensemble'; rows: metrics; first column: 'metrics'
    all_metrics = {}
    for name, preds in zip(model_names, model_preds_list_val):
        y_true_all = []
        y_pred_all = []
        for task in tasks:
            val_split = per_task_val_splits[task]
            if val_split.empty or preds[task].empty:
                continue
            pred_vals = preds[task][task]
            gt_vals = val_split[task]
            mask = (~pd.isna(pred_vals)) & (~pd.isna(gt_vals))
            y_true_all.extend(gt_vals[mask].astype(int).tolist())
            y_pred_all.extend(pred_vals[mask].astype(int).tolist())
        all_metrics[name] = compute_all_metrics(
            pd.DataFrame({'All': y_pred_all}),
            pd.DataFrame({'All': y_true_all}),
            task="All"
        )
    # Ensemble metrics
    y_true_all = []
    y_pred_all = []
    for task in tasks:
        val_split = per_task_val_splits[task]
        if val_split.empty or ensemble_preds_val.empty:
            continue
        pred_vals = ensemble_preds_val[task]
        gt_vals = val_split[task]
        mask = (~pd.isna(pred_vals)) & (~pd.isna(gt_vals))
        y_true_all.extend(gt_vals[mask].astype(int).tolist())
        y_pred_all.extend(pred_vals[mask].astype(int).tolist())
    all_metrics['Ensemble'] = compute_all_metrics(
        pd.DataFrame({'All': y_pred_all}),
        pd.DataFrame({'All': y_true_all}),
        task="All"
    )
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.insert(0, 'metrics', comparison_df.index)
    comparison_df = comparison_df.reset_index(drop=True)
    save_csv_and_latex(comparison_df, OUTPUT_COMPARISON_TABLE_PATH, latex_caption="Comparison table for all models and ensemble (All tasks)")

    print(f"Saved ensemble comparison table to {OUTPUT_COMPARISON_TABLE_PATH} and LaTeX table.")
    # 8. For separate validation set: process all images in batches, per model folder
    print("Evaluating ensemble on separate validation set (memory efficient, per-folder batching)...")
    image_files = glob.glob(os.path.join(SEPARATE_VALIDATION_DIR, "*.nii.gz"))
    image_filenames = [os.path.basename(p) for p in image_files]
    num_images = len(image_files)
    batch_size = INFERENCE_BATCH_SIZE
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    all_model_probs = []
    for idx, folder_info in enumerate(MODEL_FOLDERS):
        folder = folder_info["folder"]
        model_type = folder_info.get("model_type", "focalemd")
        print(f"Loading models from folder {idx+1}/{len(MODEL_FOLDERS)}: {folder} (type: {model_type})")
        if model_type == "classical":
            mask_dir = "data/LISA2025/Task_1_Validation_masks/"  # Adjust as needed
            classical_models = load_classical_models(folder, tasks)
            # batch_predict_classical expects a list of image paths
            df_pred = batch_predict_classical(classical_models, image_files, mask_dir, folder, tasks)
            # Convert predictions to one-hot probability arrays for each task
            model_probs = {task: [] for task in tasks}
            for i, row in df_pred.iterrows():
                for task in tasks:
                    pred = row[task]
                    if pred is not None and not pd.isna(pred):
                        one_hot = np.zeros(3)
                        one_hot[int(pred)] = 1.0
                    else:
                        one_hot = np.array([np.nan, np.nan, np.nan])
                    model_probs[task].append(one_hot)
            for task in tasks:
                model_probs[task] = np.stack(model_probs[task], axis=0)  # shape (num_images, 3)
            all_model_probs.append(model_probs)
            del classical_models
            import torch
            torch.cuda.empty_cache()
            print(f"Released classical models from {folder} and cleared GPU memory.")
        else:
            model_dict = load_model(folder, use_bayesian=USE_BAYESIAN_NETWORK, model_type=model_type, device=device)
            model_probs = {task: [] for task in tasks}
            for batch_start in range(0, num_images, batch_size):
                batch_end = min(batch_start + batch_size, num_images)
                batch_paths = image_files[batch_start:batch_end]
                # Predict probabilities for this batch
                batch_probs = predict_image_probabilities_batch(model_dict, batch_paths, transforms, device)
                for task in tasks:
                    if task in batch_probs:
                        model_probs[task].append(batch_probs[task])
                    else:
                        model_probs[task].append(np.full((batch_end-batch_start, 3), np.nan))
            for task in tasks:
                model_probs[task] = np.concatenate(model_probs[task], axis=0)
            all_model_probs.append(model_probs)
            del model_dict
            import torch
            torch.cuda.empty_cache()
            print(f"Released models from {folder} and cleared GPU memory.")
    # Use only selected models for ensemble
    selected_all_model_probs = [all_model_probs[i] for i in USE_FOR_ENSEMBLE]
    sep_val_results = ensemble_predict_from_probabilities(
        selected_all_model_probs, tasks, image_filenames,
        use_bn=USE_BAYESIAN_NETWORK, bn=bn, bn_params=bn_params, binary_collapse=BINARY_COLLAPSE
    )
    print("Finished evaluating ensemble on separate validation set.")
    sep_val_df = sep_val_results
    for task in tasks:
        if task not in sep_val_df.columns:
            sep_val_df[task] = None
    column_order = ["filename"] + tasks
    sep_val_df = sep_val_df[column_order]
    sep_val_df = sep_val_df.sort_values(by="filename").reset_index(drop=True)
    save_csv_and_latex(sep_val_df, OUTPUT_CSV_PATH, latex_caption="Ensemble predictions for separate validation set.")
    print(f"Saved ensemble predictions for separate validation set to {OUTPUT_CSV_PATH} and LaTeX table.")

def get_inference_transforms(spatial_size=(150, 150, 150)):
    """
    Returns the MONAI Compose transform for inference (no augmentation).
    """
    # Compose a set of transforms for preprocessing images before inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
        CenterSpatialCropd, SpatialPadd, ToTensord
    )
    transforms = [
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
        ToTensord(keys=["img"], dtype=torch.float32),
    ]
    return Compose(transforms)


def predict_image_probabilities(models, image_path, transforms, device="cuda"):
    """
    Given a dict of models (for all tasks), an image path, and transforms,
    returns a dict {task: np.array(probabilities)} for all tasks.
    """
    # Predicts class probabilities for a single image for all tasks
    data_dict = {"img": image_path}
    processed = transforms(data_dict)
    image_tensor = processed["img"].unsqueeze(0).to(device)
    task_probs = {}
    for task, model in models.items():
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            task_probs[task] = probs
    return task_probs

def predict_image_probabilities_batch(models, image_paths, transforms, device="cuda"):
    """
    Given a dict of models (for all tasks), a list of image paths, and transforms,
    returns a dict {task: np.array of shape (batch_size, num_classes)} for all tasks.
    """
    # Predicts class probabilities for a batch of images for all tasks
    batch_data = [{"img": path} for path in image_paths]
    processed = [transforms(d) for d in batch_data]
    image_tensors = [d["img"] for d in processed]  # each is (C, H, W, D)
    image_tensor = torch.stack(image_tensors, dim=0).to(device)  # (B, C, H, W, D)
    task_probs = {}
    for task, model in models.items():
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape (B, num_classes)
            task_probs[task] = probs
    return task_probs

def ensemble_task_probabilities(prob_list):
    """
    Given a list of probability vectors (from all ensemble members) for a single task,
    returns the ensembled probability vector (by averaging).
    """
    # Average probabilities across ensemble members for a single task
    return np.mean(np.stack(prob_list, axis=0), axis=0)


def run_ensemble_predictions(model_dicts, df, transforms, device="cuda", use_bn=False, bn=None, bn_params=None, tasks=None, batch_size=8):
    """
    Given a list of model dicts (ensemble), a DataFrame of images, and transforms,
    returns a DataFrame of ensembled predictions for the specified tasks and images.
    If use_bn is True, adjusts the ensembled probabilities with the Bayesian Network.
    bn_params: dict with keys 'epsilon', 'confidence_threshold', 'alpha'
    tasks: list of tasks to predict (default: all tasks)
    batch_size: batch size for inference
    """
    # Runs ensemble prediction for a batch of images and tasks, optionally with Bayesian Network adjustment
    if tasks is None:
        tasks = [
            "Noise", "Zipper", "Positioning", "Banding",
            "Motion", "Contrast", "Distortion"
        ]
    results = []
    num_images = len(df)
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_rows = df.iloc[batch_start:batch_end]
        image_paths = batch_rows["filename"].tolist()
        # For each task, collect probability vectors from all ensemble members
        # For each model in the ensemble, get batch probabilities for this batch
        all_model_task_probs = []  # List of dicts: [{task: (B, num_classes)}, ...]
        for model_dict in model_dicts:
            model_task_probs = predict_image_probabilities_batch({t: model_dict[t] for t in tasks if t in model_dict}, image_paths, transforms, device)
            all_model_task_probs.append(model_task_probs)
        # For each image in batch, build result
        for i, row in enumerate(batch_rows.itertuples(index=False)):
            result = {"filename": row.filename}
            prob_matrix = []
            for task_idx, task in enumerate(tasks):
                prob_list = []
                for model_task_probs in all_model_task_probs:
                    if task in model_task_probs:
                        prob_list.append(model_task_probs[task][i])
                if prob_list:
                    avg_probs = ensemble_task_probabilities(prob_list)
                    prob_matrix.append(avg_probs)
                else:
                    prob_matrix.append(np.array([np.nan, np.nan, np.nan]))
            prob_matrix = np.stack(prob_matrix, axis=0)  # shape (num_tasks, num_classes)
            # Optionally adjust with BN
            if use_bn and bn is not None:
                prob_matrix = adjust_probs_with_bn(
                    bn, tasks, prob_matrix,
                    epsilon=bn_params.get('epsilon', 1e-8),
                    confidence_threshold=bn_params.get('confidence_threshold', 0.6),
                    alpha=bn_params.get('alpha', 0.3)
                )
            # Take argmax for each task, or binary collapse if enabled
            for j, task in enumerate(tasks):
                if not np.isnan(prob_matrix[j]).all():
                    if BINARY_COLLAPSE:
                        result[task] = int(binary_collapse_predict(prob_matrix[j]))
                    else:
                        result[task] = int(np.nanargmax(prob_matrix[j]))
                else:
                    result[task] = None
            results.append(result)
    return pd.DataFrame(results)

def train_or_load_bayesian_network(train_csv_path, tasks, model_path=None):
    """
    Train a Bayesian Network on the full training set, or load from pickle if model_path exists.
    Returns: DiscreteBayesianNetwork
    """
    # Trains or loads a Bayesian Network for post-processing predictions
    import os
    import pandas as pd
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    import pickle
    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            bn = pickle.load(f)
        return bn
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[train_df['filename'].notna()]
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
    if model_path:
        with open(model_path, "wb") as f:
            pickle.dump(bn, f)
    return bn


def adjust_probs_with_bn(bn, tasks, orig_probs, epsilon=1e-8, confidence_threshold=0.6, alpha=0.3):
    """
    Adjusts ensembled probabilities for all tasks using the Bayesian Network (run_bn_soft_evidence_v2 logic).
    orig_probs: shape (num_tasks, num_classes)
    Returns: adjusted_probs (same shape)
    """
    # Adjusts probabilities using Bayesian Network inference and soft evidence
    import numpy as np
    from pgmpy.inference import VariableElimination
    adjusted_probs = orig_probs.copy()
    try:
        inference = VariableElimination(bn)
        for i, target_task in enumerate(tasks):
            try:
                # Collect evidence from other confident predictions
                evidence = {}
                for j, other_task in enumerate(tasks):
                    if i != j:
                        max_prob = np.max(orig_probs[j])
                        if max_prob > confidence_threshold:
                            predicted_class = np.argmax(orig_probs[j])
                            evidence[other_task] = predicted_class
                if evidence:
                    result = inference.query(variables=[target_task], evidence=evidence)
                    bn_conditional = result.values
                else:
                    result = inference.query(variables=[target_task])
                    bn_conditional = result.values
                # Simple mixing
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

def save_csv_and_latex(df, csv_path, latex_caption="Placeholder caption."):
    """
    Save a DataFrame as CSV and as a LaTeX table (booktabs, 2 decimals, ready for publication).
    """
    # Mapping for human-friendly metric names
    metric_name_map = {
        "accuracy": "Accuracy (Micro)",
        "accuracy_micro": "Accuracy (Micro)",
        "accuracy_macro": "Accuracy (Macro)",
        "accuracy_weighted": "Accuracy (Weighted)",
        "f1_micro": "F1 (Micro)",
        "f1_macro": "F1 (Macro)",
        "f1_weighted": "F1 (Weighted)",
        "f2_micro": "F2 (Micro)",
        "f2_macro": "F2 (Macro)",
        "f2_weighted": "F2 (Weighted)",
        "precision_micro": "Precision (Micro)",
        "precision_macro": "Precision (Macro)",
        "precision_weighted": "Precision (Weighted)",
        "recall_micro": "Recall (Micro)",
        "recall_macro": "Recall (Macro)",
        "recall_weighted": "Recall (Weighted)",
        "Composite": "Composite",
    }
    # If 'metrics' is a column, map its values
    if 'metrics' in df.columns:
        df['metrics'] = df['metrics'].map(lambda x: metric_name_map.get(x, x))
    # If index is metrics, map index
    if df.index.name == 'metrics' or (df.index.nlevels == 1 and any(k in metric_name_map for k in df.index)):
        df = df.rename(index=metric_name_map)
    df.to_csv(csv_path, index=False)
    latex_path = csv_path.rsplit('.', 1)[0] + '.tex'
    # Format all floats to 2 decimals
    df_fmt = df.copy()
    for col in df_fmt.select_dtypes(include=['float', 'float64', 'float32']):
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    # Map metrics in LaTeX as well
    if 'metrics' in df_fmt.columns:
        df_fmt['metrics'] = df_fmt['metrics'].map(lambda x: metric_name_map.get(x, x))
    if df_fmt.index.name == 'metrics' or (df_fmt.index.nlevels == 1 and any(k in metric_name_map for k in df_fmt.index)):
        df_fmt = df_fmt.rename(index=metric_name_map)
    latex_str = df_fmt.to_latex(index=False, float_format="{:.3f}".format, caption=latex_caption, label=None, escape=False, longtable=False, bold_rows=False, column_format=None, multicolumn=True, multicolumn_format='c', multirow=False, na_rep="")
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {latex_path}")

def binary_collapse_predict(prob_vector):
    """
    Implements the binary collapse strategy:
    If p(0) > p(1) + p(2), predict 0. Else, predict 1 if p(1) > p(2) else 2.
    """
    if prob_vector[0] > prob_vector[1] + prob_vector[2]:
        return 0
    else:
        return 1 if prob_vector[1] > prob_vector[2] else 2

def get_mask_path(mask_dir, image_filename):
    """
    Given the mask directory and image filename, construct the mask path.
    For LISA_VALIDATION_XXXX_LF_{axi,cor,sag}.nii.gz, mask is
    LISA_VALIDATION_XXXX_LF_{axi,cor,sag}_mask.nii.gz in mask_dir.
    Returns None if not a .nii.gz file.
    """
    import os
    basename = os.path.basename(image_filename)
    if not basename.endswith('.nii.gz'):
        print(f"Warning: Skipping non-NIfTI file: {basename}")
        return None
    mask_name = basename[:-7] + '_mask.nii.gz'
    mask_path = os.path.join(mask_dir, mask_name)
    return mask_path

def extract_features(image_path, mask_path):
    """
    Extract features from a single image and mask using NeonatalMRIQualityAssessment.
    """
    try:
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(int)
    except Exception as e:
        print(f"Error loading mask {mask_path}: {str(e)}")
        return None
    try:
        qa = NeonatalMRIQualityAssessment(image_path, mask_data)
        features = qa.extract_all_quality_markers()
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

def load_classical_models(model_folder, tasks, model_types=["rf", "xgb"]):
    """
    Loads classical models (rf, xgb) for each task from the specified folder.
    Returns a dict: {task: {model_type: model}}
    """
    import os
    models = {task: {} for task in tasks}
    for task in tasks:
        for model_type in model_types:
            model_path = os.path.join(model_folder, f"{task}_{model_type}.pkl")
            if os.path.exists(model_path):
                try:
                    models[task][model_type] = joblib.load(model_path)
                except Exception as e:
                    print(f"Error loading {model_type} model for {task} in {model_folder}: {e}")
            else:
                print(f"{model_type} model checkpoint not found for {task} in {model_folder}")
    return models

def run_bet(input_file, output_file):
    """
    Run FSL's BET tool to create brain mask.
    Args:
        input_file (str): Path to input NIfTI file
        output_file (str): Path to output mask file (without _mask.nii.gz)
    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import subprocess
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd = ["bet", input_file, output_file, "-m", "-n", "-f", "0.2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running BET on {input_file}: {result.stderr}")
        return False
    return True

def batch_predict_classical(models, image_paths, mask_dir, model_dir, tasks):
    """
    Classical model prediction with robust feature handling (matches validation_classical.py):
    - For each image, if a features JSON exists, load features from it.
    - Otherwise, extract features, save as JSON, and use them.
    - Drop columns with NaN/inf values
    - Load per-task feature lists from JSON
    - For each prediction, use per-task feature list if available, otherwise all valid features
    - Use model's feature_names_in_ for ordering, filling missing features with np.nan
    """
    import numpy as np
    import pandas as pd
    import os
    features_list = []
    filenames = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        # Check for pre-extracted features JSON first
        scan_dir = os.path.dirname(image_path)
        features_dir = os.path.join(scan_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_file = os.path.join(features_dir, filename.replace('.nii.gz', '.json'))
        features = None
        if os.path.exists(features_file):
            try:
                with open(features_file, 'r') as f:
                    features_data = json.load(f)
                features = features_data['features'] if 'features' in features_data else features_data
            except Exception as e:
                print(f"Error loading features from {features_file}: {e}")
                features = None
        if features is None:
            mask_path = get_mask_path(mask_dir, image_path)
            if mask_path is None:
                print(f"Mask path could not be determined for {image_path}, skipping.")
                continue
            if not os.path.exists(mask_path):
                print(f"Mask not found for {image_path}, creating with BET...")
                bet_output_prefix = mask_path[:-12]  # strip _mask.nii.gz
                success = run_bet(image_path, bet_output_prefix)
                if not success or not os.path.exists(mask_path):
                    print(f"Failed to create mask for {image_path}, skipping.")
                    continue
            features = extract_features(image_path, mask_path)
            if features is not None:
                features_data = {'features': features}
                with open(features_file, 'w') as f:
                    json.dump(features_data, f, indent=2)
        if features is None:
            continue
        features_list.append(features)
        filenames.append(filename)
    # Build DataFrame and drop columns with NaN or inf values
    if features_list:
        features_df = pd.DataFrame(features_list, index=filenames)
        exclude_columns = ['subject_id', 'data_type', 'filename'] + tasks
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        is_inf = np.isinf(features_df[feature_columns].select_dtypes(include=[np.number]))
        nan_mask = features_df[feature_columns].isna().any()
        inf_mask = is_inf.any()
        problematic_columns = features_df[feature_columns].columns[nan_mask | inf_mask]
        if len(problematic_columns) > 0:
            print(f"Dropping {len(problematic_columns)} columns with NaN or inf values:")
            print(list(problematic_columns))
            features_df = features_df.drop(columns=problematic_columns)
            feature_columns = [col for col in feature_columns if col not in problematic_columns]
        # Load feature lists for each task from JSON (if available)
        task_feature_lists = {}
        for task in tasks:
            feature_json = os.path.join(model_dir, f"{task}_rf_features.json")
            if os.path.exists(feature_json):
                with open(feature_json, 'r') as f:
                    task_feature_lists[task] = json.load(f)
            else:
                task_feature_lists[task] = None
    else:
        features_df = pd.DataFrame()
        feature_columns = []
        task_feature_lists = {task: None for task in tasks}
    # Predict
    results = []
    for filename in filenames:
        result = {"filename": filename}
        if features_df.empty or filename not in features_df.index:
            for task in tasks:
                result[task] = None
            results.append(result)
            continue
        for task in tasks:
            pred = None
            for model_type in ["rf", "xgb"]:
                if model_type in models[task]:
                    model = models[task][model_type]
                    feature_list = task_feature_lists.get(task)
                    if feature_list is not None:
                        features = features_df.loc[filename, feature_list].to_dict()
                    else:
                        features = features_df.loc[filename, feature_columns].to_dict()
                    if hasattr(model, "feature_names_in_"):
                        feature_order = model.feature_names_in_
                        feature_vec = [features.get(f, np.nan) for f in feature_order]
                    else:
                        feature_vec = [features[k] for k in sorted(features.keys())]
                    try:
                        pred = int(model.predict([feature_vec])[0])
                        break  # Use first available model type
                    except Exception as e:
                        print(f"Error predicting {task} for {filename} with {model_type}: {e}")
                        pred = None
            result[task] = pred
        results.append(result)
    df = pd.DataFrame(results)
    for task in tasks:
        if task not in df.columns:
            df[task] = None
    df = df[["filename"] + tasks]
    return df

def batch_predict_classical_proba(models, image_paths, mask_dir, model_dir, tasks):
    """
    Like batch_predict_classical, but returns per-sample, per-task probability vectors (shape (N, 3)).
    Uses predict_proba if available, otherwise one-hot encoding of predicted class.
    Returns: dict {task: list of np.array([p0, p1, p2])} (length N per task)
    """
    import numpy as np
    import os
    import json
    import pandas as pd
    features_list = []
    filenames = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        scan_dir = os.path.dirname(image_path)
        features_dir = os.path.join(scan_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_file = os.path.join(features_dir, filename.replace('.nii.gz', '.json'))
        features = None
        if os.path.exists(features_file):
            try:
                with open(features_file, 'r') as f:
                    features_data = json.load(f)
                features = features_data['features'] if 'features' in features_data else features_data
            except Exception as e:
                features = None
        if features is None:
            mask_path = get_mask_path(mask_dir, image_path)
            if mask_path is None or not os.path.exists(mask_path):
                continue
            features = extract_features(image_path, mask_path)
            if features is not None:
                features_data = {'features': features}
                with open(features_file, 'w') as f:
                    json.dump(features_data, f, indent=2)
        if features is None:
            continue
        features_list.append(features)
        filenames.append(filename)
    if features_list:
        features_df = pd.DataFrame(features_list, index=filenames)
        exclude_columns = ['subject_id', 'data_type', 'filename'] + tasks
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        is_inf = np.isinf(features_df[feature_columns].select_dtypes(include=[np.number]))
        nan_mask = features_df[feature_columns].isna().any()
        inf_mask = is_inf.any()
        problematic_columns = features_df[feature_columns].columns[nan_mask | inf_mask]
        if len(problematic_columns) > 0:
            features_df = features_df.drop(columns=problematic_columns)
            feature_columns = [col for col in feature_columns if col not in problematic_columns]
        task_feature_lists = {}
        for task in tasks:
            feature_json = os.path.join(model_dir, f"{task}_rf_features.json")
            if os.path.exists(feature_json):
                with open(feature_json, 'r') as f:
                    task_feature_lists[task] = json.load(f)
            else:
                task_feature_lists[task] = None
    else:
        features_df = pd.DataFrame()
        feature_columns = []
        task_feature_lists = {task: None for task in tasks}
    # Predict probabilities
    results = {task: [] for task in tasks}
    for filename in filenames:
        if features_df.empty or filename not in features_df.index:
            for task in tasks:
                results[task].append(np.array([np.nan, np.nan, np.nan]))
            continue
        for task in tasks:
            proba = None
            for model_type in ["rf", "xgb"]:
                if model_type in models[task]:
                    model = models[task][model_type]
                    feature_list = task_feature_lists.get(task)
                    if feature_list is not None:
                        features = features_df.loc[filename, feature_list].to_dict()
                    else:
                        features = features_df.loc[filename, feature_columns].to_dict()
                    if hasattr(model, "feature_names_in_"):
                        feature_order = model.feature_names_in_
                        feature_vec = [features.get(f, np.nan) for f in feature_order]
                    else:
                        feature_vec = [features[k] for k in sorted(features.keys())]
                    try:
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba([feature_vec])[0]
                            # Ensure length 3
                            if len(proba) < 3:
                                # Fill missing classes with 0
                                full_proba = np.zeros(3)
                                for idx, c in enumerate(model.classes_):
                                    full_proba[int(c)] = proba[idx]
                                proba = full_proba
                        else:
                            pred = int(model.predict([feature_vec])[0])
                            proba = np.zeros(3)
                            proba[pred] = 1.0
                        break
                    except Exception as e:
                        proba = np.array([np.nan, np.nan, np.nan])
            if proba is None:
                proba = np.array([np.nan, np.nan, np.nan])
            results[task].append(proba)
    return results, filenames

def ensemble_predict_from_probabilities(
    all_model_probs,  # list of dicts: {task: [np.array([p0, p1, p2]), ...]}
    tasks,            # list of task names
    filenames,        # list of filenames (samples)
    use_bn=False,
    bn=None,
    bn_params=None,
    binary_collapse=False
):
    """
    Given per-model per-task per-sample probabilities, ensemble by averaging, optional BN, and binary collapse/argmax.
    Returns DataFrame: columns = ["filename"] + tasks, rows = samples.
    """
    import numpy as np
    import pandas as pd
    if bn_params is None:
        bn_params = dict(epsilon=1e-8, confidence_threshold=0.6, alpha=0.3)
    num_samples = len(filenames)
    results = []
    for i in range(num_samples):
        result = {"filename": filenames[i]}
        # Build prob_matrix: shape (num_tasks, num_classes)
        prob_matrix = []
        for t in tasks:
            prob_list = [model_probs[t][i] for model_probs in all_model_probs if i < len(model_probs[t]) and model_probs[t][i] is not None and not np.isnan(model_probs[t][i]).all()]
            if prob_list:
                avg_prob = ensemble_task_probabilities(prob_list)
            else:
                avg_prob = np.array([np.nan, np.nan, np.nan])
            prob_matrix.append(avg_prob)
        prob_matrix = np.stack(prob_matrix, axis=0)
        # Optionally adjust with BN
        if use_bn and bn is not None:
            prob_matrix = adjust_probs_with_bn(
                bn, tasks, prob_matrix,
                epsilon=bn_params.get('epsilon', 1e-8),
                confidence_threshold=bn_params.get('confidence_threshold', 0.6),
                alpha=bn_params.get('alpha', 0.3)
            )
        # For each task, apply binary collapse or argmax
        for j, t in enumerate(tasks):
            p = prob_matrix[j]
            if not np.isnan(p).all():
                if binary_collapse:
                    result[t] = int(binary_collapse_predict(p))
                else:
                    result[t] = int(np.nanargmax(p))
            else:
                result[t] = None
        results.append(result)
    df = pd.DataFrame(results)
    column_order = ["filename"] + tasks
    df = df[column_order]
    return df

# =========================
# End of Script
# =========================

if __name__ == "__main__":
    # Entry point for running the ensemble evaluation as a script
    print("Starting ensemble evaluation...")
    run_ensemble_evaluation()
    print("Ensemble evaluation completed.")
