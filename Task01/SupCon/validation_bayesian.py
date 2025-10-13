"""
10-fold cross-validated Bayesian Network hybrid inference script
for LISA 2025 QC data.
Loads 7 pre-trained models, runs inference on raw BIDS data, applies
Bayesian adjustment using a learned Bayesian Network (BN) structure,
and saves both original and adjusted probabilities.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    CenterSpatialCropd, SpatialPadd, ToTensord
)
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
import pickle

# Import model classes
from model import (
    OrdinalContrastiveLightningModule,
    OrdinalFocalEMDLightningModule
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="10-fold Bayesian Network hybrid validation inference"
    )
    parser.add_argument(
        "--input_csv", type=str,
        default="data/LISA2025/BIDS/LISA_2025_bids.csv",
        help="CSV with raw BIDS data and labels"
    )
    parser.add_argument(
        "--model_dir", type=str, default="results_emd_augmented_onlyemd_final/",
        help="Directory with trained models"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results_emd_augmented_onlyemd_final/",
        help="Directory to save output CSVs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for inference"
    )
    parser.add_argument(
        "--spatial_size", type=int, nargs=3, default=[150, 150, 150],
        help="Spatial size for image processing"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["contrastive", "focalemd"],
        default="focalemd", help="Model type to use"
    )
    parser.add_argument(
        "--num_folds", type=int, default=10,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Epsilon for numerical stability in probabilities"
    )
    return parser.parse_args()

def get_inference_transforms(spatial_size):
    return Compose([
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
        ToTensord(keys=["img"], dtype=torch.float32),
    ])

def load_models(model_dir, model_type, device, tasks):
    model_class = (
        OrdinalContrastiveLightningModule
        if model_type == "contrastive" else OrdinalFocalEMDLightningModule
    )
    models = {}
    for task in tasks:
        model_path = os.path.join(model_dir, f"{task}_finalmodel.ckpt")
        if os.path.exists(model_path):
            models[task] = (
                model_class.load_from_checkpoint(model_path)
                .to(device)
            )
            models[task].eval()
        else:
            print(
                f"Warning: Model not found for {task} at {model_path}. Skipping."
            )
    return models

def predict_image(models, image_path, transforms, device, tasks):
    # Returns: (probs: np.ndarray shape (7, 3))
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

def run_bn_soft_evidence(bn, cpts, tasks, orig_probs, epsilon=1e-8):
    # orig_probs: shape (7,3), returns adjusted_probs: shape (7,3)
    # Use BeliefPropagation with soft evidence
    from pgmpy.inference import BeliefPropagation
    # Set up soft evidence as virtual evidence
    evidence = {}
    for i, task in enumerate(tasks):
        # Epsilon-clipped
        p = np.clip(orig_probs[i], epsilon, 1.0)
        p = p / p.sum()  # Renormalize
        evidence[task] = p
    # BeliefPropagation with soft evidence
    bp = BeliefPropagation(bn)
    marginals = {}
    for i, task in enumerate(tasks):
        try:
            q = bp.query(variables=[task], virtual_evidence={task: evidence[task]})
            marginals[task] = q[task].values
        except Exception as e:
            print(f"BN inference error for {task}: {e}")
            marginals[task] = np.full(3, np.nan)
    # Stack into array
    adjusted_probs = np.stack([marginals[t] for t in tasks], axis=0)
    return adjusted_probs


# Alternative version using hard evidence from confident predictions
def run_bn_soft_evidence_v2(bn, cpts, tasks, orig_probs, epsilon=1e-8, confidence_threshold=0.6):
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
    
    # Validate inputs
    if orig_probs is None or np.isnan(orig_probs).all():
        print("Warning: All original probabilities are NaN, returning as-is")
        return orig_probs

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
                    result = inference.query(variables=[target_task], evidence=evidence)
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




def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    # Load data
    df = pd.read_csv(args.input_csv)
    df = df[df['filename'].notna()]
    if 'sub' not in df.columns:
        df['sub'] = df['filename'].apply(
            lambda x: os.path.basename(x).split('_')[0]
        )
    label_cols = tasks
    y = df[label_cols].fillna(0).astype(int).sum(axis=1)
    groups = df['sub']
    sgkf = StratifiedGroupKFold(
        n_splits=args.num_folds, shuffle=True, random_state=args.random_seed
    )
    models = load_models(
        args.model_dir, args.model_type, args.device, tasks
    )
    transforms = get_inference_transforms(tuple(args.spatial_size))
    orig_rows = []
    adj_rows = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
        print(f"Fold {fold+1}/{args.num_folds}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
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
        train_labels = train_df[label_cols].fillna(0).astype(int)
        bn.fit(train_labels, estimator=MaximumLikelihoodEstimator)
        # Save BN structure and parameters
        bn_path = os.path.join(
            args.output_dir, f"bn_fold{fold}.pgmpy"
        )
        with open(bn_path, "wb") as f:
            pickle.dump(bn, f)
        print(f"Saved BN for fold {fold} to {bn_path}")
        # For each image in val, run inference and adjust
        for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
            image_path = row['filename']
            subject = row['sub']
            true_labels = [row[t] if t in row else np.nan for t in tasks]
            try:
                orig_probs = predict_image(
                    models, image_path, transforms, args.device, tasks
                )  # (7,3)
                adj_probs = run_bn_soft_evidence_v2(
                    bn, None, tasks, orig_probs, epsilon=args.epsilon
                )
                
                # Check if adjusted probabilities have NaN values
                if np.isnan(adj_probs).any() or np.isnan(adj_probs).all():
                    print(f"Warning: Adjusted probabilities contain NaN for {image_path}, using original probabilities")
                    adj_probs = orig_probs.copy()
                
                orig_pred = np.nanargmax(orig_probs, axis=1)
                adj_pred = np.nanargmax(adj_probs, axis=1)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Try to get original probabilities even if adjustment fails
                try:
                    orig_probs = predict_image(
                        models, image_path, transforms, args.device, tasks
                    )
                    adj_probs = orig_probs.copy()  # Use original as fallback
                    orig_pred = np.nanargmax(orig_probs, axis=1)
                    adj_pred = orig_pred.copy()  # Use original predictions as fallback
                except Exception as e2:
                    print(f"Failed to get original probabilities for {image_path}: {e2}")
                    orig_probs = np.full((7, 3), np.nan)
                    adj_probs = np.full((7, 3), np.nan)
                    orig_pred = [np.nan] * 7
                    adj_pred = [np.nan] * 7
            orig_rows.append({
                'fold': fold,
                'filename': image_path,
                'subject': subject,
                **{f"true_{t}": l for t, l in zip(tasks, true_labels)},
                **{
                    f"orig_prob_{t}_{k}": orig_probs[i, k]
                    for i, t in enumerate(tasks)
                    for k in range(3)
                },
                **{
                    f"orig_pred_{t}": orig_pred[i]
                    for i, t in enumerate(tasks)
                }
            })
            adj_rows.append({
                'fold': fold,
                'filename': image_path,
                'subject': subject,
                **{f"true_{t}": l for t, l in zip(tasks, true_labels)},
                **{
                    f"adj_prob_{t}_{k}": adj_probs[i, k]
                    for i, t in enumerate(tasks)
                    for k in range(3)
                },
                **{
                    f"adj_pred_{t}": adj_pred[i]
                    for i, t in enumerate(tasks)
                }
            })
    # Save to CSV
    orig_df = pd.DataFrame(orig_rows)
    adj_df = pd.DataFrame(adj_rows)
    orig_csv = os.path.join(args.output_dir, "cv_original_probs.csv")
    adj_csv = os.path.join(args.output_dir, "cv_adjusted_probs.csv")
    orig_df.to_csv(orig_csv, index=False)
    adj_df.to_csv(adj_csv, index=False)
    print(f"Saved original probabilities to {orig_csv}")
    print(f"Saved adjusted probabilities to {adj_csv}")
    # Compute summary statistics (mean, std) for each prob/pred column
    def add_summary(df, prefix):
        means = df.mean(numeric_only=True)
        stds = df.std(numeric_only=True)
        summary = pd.DataFrame(
            [means, stds], index=[f"{prefix}_mean", f"{prefix}_std"]
        )
        return pd.concat([df, summary], axis=0)
    orig_df = add_summary(orig_df, "orig")
    adj_df = add_summary(adj_df, "adj")
    orig_df.to_csv(orig_csv, index=False)
    adj_df.to_csv(adj_csv, index=False)
    print("Appended summary statistics to output CSVs.")
    # Compute and output metrics for both original and adjusted predictions
    from sklearn.metrics import (
        f1_score, fbeta_score, precision_score, recall_score, accuracy_score
    )
    metrics = [
        ("accuracy", accuracy_score),
        ("f1_micro", lambda y, p: f1_score(y, p, average="micro")),
        ("f1_macro", lambda y, p: f1_score(y, p, average="macro")),
        ("f1_weighted", lambda y, p: f1_score(y, p, average="weighted")),
        ("f2_micro", lambda y, p: fbeta_score(y, p, beta=2, average="micro")),
        ("f2_macro", lambda y, p: fbeta_score(y, p, beta=2, average="macro")),
        ("f2_weighted", lambda y, p: fbeta_score(y, p, beta=2, average="weighted")),
        ("precision_micro", lambda y, p: precision_score(y, p, average="micro")),
        ("precision_macro", lambda y, p: precision_score(y, p, average="macro")),
        ("precision_weighted", lambda y, p: precision_score(y, p, average="weighted")),
        ("recall_micro", lambda y, p: recall_score(y, p, average="micro")),
        ("recall_macro", lambda y, p: recall_score(y, p, average="macro")),
        ("recall_weighted", lambda y, p: recall_score(y, p, average="weighted")),
    ]
    def compute_metrics_table(df, pred_prefix, tasks):
        table = pd.DataFrame(index=[m[0] for m in metrics], columns=tasks)
        all_y_true = []
        all_y_pred = []
        for task in tasks:
            y_true = df[f"true_{task}"].values
            y_pred = df[f"{pred_prefix}_pred_{task}"].values
            mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
            y_true = y_true[mask].astype(int)
            y_pred = y_pred[mask].astype(int)
            if len(y_true) > 0:
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
            for mname, mfunc in metrics:
                if len(y_true) == 0:
                    val = float('nan')
                else:
                    try:
                        val = mfunc(y_true, y_pred)
                    except Exception:
                        val = float('nan')
                table.at[mname, task] = val
        # Overall metrics
        if all_y_true and all_y_pred:
            all_y_true_concat = np.concatenate(all_y_true)
            all_y_pred_concat = np.concatenate(all_y_pred)
        else:
            all_y_true_concat = np.array([])
            all_y_pred_concat = np.array([])
        all_col = []
        for mname, mfunc in metrics:
            if len(all_y_true_concat) == 0:
                val = float('nan')
            else:
                try:
                    val = mfunc(all_y_true_concat, all_y_pred_concat)
                except Exception:
                    val = float('nan')
            all_col.append(val)
        table["All"] = all_col
        cols = ["All"] + [c for c in table.columns if c != "All"]
        table = table[cols]
        return table
    orig_df = pd.read_csv(orig_csv)
    adj_df = pd.read_csv(adj_csv)
    orig_df = orig_df[~orig_df['fold'].astype(str).str.contains('orig_')]
    adj_df = adj_df[~adj_df['fold'].astype(str).str.contains('adj_')]
    orig_metrics = compute_metrics_table(orig_df, "orig", tasks)
    adj_metrics = compute_metrics_table(adj_df, "adj", tasks)
    orig_metrics_csv = os.path.join(
        args.output_dir, "cv_original_metrics.csv"
    )
    adj_metrics_csv = os.path.join(
        args.output_dir, "cv_adjusted_metrics.csv"
    )
    orig_metrics.to_csv(orig_metrics_csv)
    adj_metrics.to_csv(adj_metrics_csv)
    print(f"Saved original metrics to {orig_metrics_csv}")
    print(f"Saved adjusted metrics to {adj_metrics_csv}")
    print("\nOriginal metrics:")
    print(orig_metrics.round(3))
    print("\nAdjusted metrics:")
    print(adj_metrics.round(3))

if __name__ == "__main__":
    main() 