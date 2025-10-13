import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import StratifiedGroupKFold
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)
from model import OrdinalContrastiveLightningModule
from model import OrdinalFocalEMDLightningModule


def get_inference_transforms(spatial_size=(128, 128, 128)):
    return Compose([
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
        ToTensord(keys=["img"], dtype=torch.float32),
    ])


def load_model(model_path, model_class, device="cuda"):
    if not os.path.exists(model_path):
        return None
    model = model_class.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    return model


def predict(model, image_path, transforms, device="cuda"):
    data_dict = {"img": image_path}
    processed = transforms(data_dict)
    image_tensor = processed["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on raw images and output metrics "
        "table."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results_emd_augmented_online",
        help="Directory with trained models"
    )
    parser.add_argument(
        "--raw_csv",
        type=str,
        default="data/LISA2025/BIDS_norm/LISA_2025_bids.csv",
        help="CSV with raw image labels"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_emd_augmented_online/"
                "raw_eval_metrics.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--output_latex",
        type=str,
        default="results_emd_augmented_online/"
                "raw_eval_metrics.tex",
        help="Output LaTeX path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference"
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
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for validation split (must match training)"
    )
    args = parser.parse_args()

    if args.model_type == "contrastive":
        model_class = OrdinalContrastiveLightningModule
    elif args.model_type == "focalemd":
        model_class = OrdinalFocalEMDLightningModule
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
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

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.read_csv(args.raw_csv)
    transforms = get_inference_transforms(tuple(args.spatial_size))

    results = {task: {"y_true": [], "y_pred": []} for task in tasks}
    models = {}
    for task in tasks:
        model_path = os.path.join(
            args.model_dir,
            f"{task}_finalmodel.ckpt"
        )
        model = load_model(model_path, model_class, args.device)
        if model is None:
            print(
                f"Warning: Model not found for {task} at {model_path}. "
                "Skipping."
            )
        models[task] = model

    for task in tasks:
        model = models[task]
        if model is None:
            continue
        # Only use rows with valid label (0,1,2) and valid filename
        mask = df[task].isin([0, 1, 2]) & df["filename"].notna()
        subdf = df[mask]
        if subdf.empty:
            print(f"No valid samples for {task}")
            continue
        y_true = subdf[task].astype(int).values
        y_pred = []
        for fname in subdf["filename"].values:
            try:
                pred = predict(model, fname, transforms, args.device)
            except Exception as e:
                print(f"Error processing {fname} for {task}: {e}")
                pred = -1
            y_pred.append(pred)
        # Remove failed predictions
        valid = np.array(y_pred) != -1
        results[task]["y_true"] = y_true[valid]
        results[task]["y_pred"] = np.array(y_pred)[valid]

    # Compute metrics
    table = pd.DataFrame(index=[m[0] for m in metrics], columns=tasks)
    # Gather all y_true and y_pred for combined metrics
    all_y_true = []
    all_y_pred = []
    for task in tasks:
        y_true = results[task]["y_true"]
        y_pred = results[task]["y_pred"]
        if len(y_true) > 0:
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
    if all_y_true and all_y_pred:
        all_y_true_concat = np.concatenate(all_y_true)
        all_y_pred_concat = np.concatenate(all_y_pred)
    else:
        all_y_true_concat = np.array([])
        all_y_pred_concat = np.array([])

    for task in tasks:
        y_true = results[task]["y_true"]
        y_pred = results[task]["y_pred"]
        if len(y_true) == 0:
            table[task] = "--"
            continue
        for mname, mfunc in metrics:
            try:
                val = mfunc(y_true, y_pred)
            except Exception:
                val = np.nan
            table.at[mname, task] = (
                f"{val:.3f}" if not np.isnan(val) else "--"
            )

    # Compute metrics for all tasks combined
    all_col = []
    for mname, mfunc in metrics:
        if len(all_y_true_concat) == 0:
            val = np.nan
        else:
            try:
                val = mfunc(all_y_true_concat, all_y_pred_concat)
            except Exception:
                val = np.nan
        all_col.append(f"{val:.3f}" if not np.isnan(val) else "--")
    table["All"] = all_col
    # Move 'All' column to the first position for clarity
    cols = ["All"] + [c for c in table.columns if c != "All"]
    table = table[cols]

    table.to_csv(args.output_csv)
    # LaTeX output
    with open(args.output_latex, "w") as f:
        f.write(
            table.to_latex(
                escape=False,
                column_format='l' + 'c'*len(table.columns),
                bold_rows=True,
                caption="Evaluation metrics on raw images (overall, per task)",
                label="tab:raw_eval_metrics",
                longtable=False,
                multicolumn=True,
                multicolumn_format='c',
                na_rep='--',
                index=True,
                float_format=".3f",
                buf=None,
            )
        )
    print(f"Saved metrics to {args.output_csv} and {args.output_latex}")

    # --- After main evaluation, repeat for validation set only ---
    # Extract subject IDs from filenames (must match train_emd.py logic)
    df_valsplit = df.copy()
    if "filename" in df_valsplit.columns:
        df_valsplit = df_valsplit[df_valsplit["filename"].notna()]
        filenames = df_valsplit["filename"].astype(str).values
        # Subject is first part of basename, split by '_', e.g. sub-XXXX
        subjects = [os.path.basename(f).split('_')[0] for f in filenames]
        df_valsplit["subject"] = subjects
    else:
        print("No 'filename' column found in CSV. Skipping validation split evaluation.")
        return
    # For validation split, use only rows with at least one valid label (0,1,2) in any task
    tasks = [
        "Noise", "Zipper", "Positioning", "Banding",
        "Motion", "Contrast", "Distortion"
    ]
    # For split, use only rows with at least one valid label in any task
    mask_any_valid = df_valsplit[tasks].apply(lambda row: any(x in [0,1,2] for x in row), axis=1)
    df_valsplit = df_valsplit[mask_any_valid]
    # For split, use the first task (Noise) as label for stratification (as in train_emd.py)
    # But to match train_emd.py, we should do the split per task, so we will do the split for each task and take the union of all validation subjects
    val_subjects_set = set()
    for task in tasks:
        # Only use rows with valid label for this task
        mask = df_valsplit[task].isin([0, 1, 2])
        subdf = df_valsplit[mask]
        if subdf.empty:
            continue
        labels = subdf[task].astype(int).values
        filenames = subdf["filename"].astype(str).values
        subjects = subdf["subject"].values
        # StratifiedGroupKFold as in train_emd.py
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        try:
            train_idx, val_idx = next(sgkf.split(filenames, labels, subjects))
            val_subjects = set(subjects[val_idx])
            val_subjects_set.update(val_subjects)
        except Exception as e:
            print(f"Error in validation split for task {task}: {e}")
            continue
    # Now filter the original df to only those validation subjects
    df_valonly = df[df["filename"].notna()].copy()
    df_valonly["subject"] = [os.path.basename(f).split('_')[0] for f in df_valonly["filename"].astype(str).values]
    df_valonly = df_valonly[df_valonly["subject"].isin(val_subjects_set)]
    # Now repeat the evaluation logic, but only for df_valonly
    results_val = {task: {"y_true": [], "y_pred": []} for task in tasks}
    for task in tasks:
        model = models[task]
        if model is None:
            continue
        mask = df_valonly[task].isin([0, 1, 2]) & df_valonly["filename"].notna()
        subdf = df_valonly[mask]
        if subdf.empty:
            continue
        y_true = subdf[task].astype(int).values
        y_pred = []
        for fname in subdf["filename"].values:
            try:
                pred = predict(model, fname, transforms, args.device)
            except Exception as e:
                pred = -1
            y_pred.append(pred)
        valid = np.array(y_pred) != -1
        results_val[task]["y_true"] = y_true[valid]
        results_val[task]["y_pred"] = np.array(y_pred)[valid]
    # Compute metrics for validation set
    table_val = pd.DataFrame(index=[m[0] for m in metrics], columns=tasks)
    all_y_true_val = []
    all_y_pred_val = []
    for task in tasks:
        y_true = results_val[task]["y_true"]
        y_pred = results_val[task]["y_pred"]
        if len(y_true) > 0:
            all_y_true_val.append(y_true)
            all_y_pred_val.append(y_pred)
    if all_y_true_val and all_y_pred_val:
        all_y_true_concat_val = np.concatenate(all_y_true_val)
        all_y_pred_concat_val = np.concatenate(all_y_pred_val)
    else:
        all_y_true_concat_val = np.array([])
        all_y_pred_concat_val = np.array([])
    for task in tasks:
        y_true = results_val[task]["y_true"]
        y_pred = results_val[task]["y_pred"]
        if len(y_true) == 0:
            table_val[task] = "--"
            continue
        for mname, mfunc in metrics:
            try:
                val = mfunc(y_true, y_pred)
            except Exception:
                val = np.nan
            table_val.at[mname, task] = (
                f"{val:.3f}" if not np.isnan(val) else "--"
            )
    # Compute metrics for all tasks combined (validation set)
    all_col_val = []
    for mname, mfunc in metrics:
        if len(all_y_true_concat_val) == 0:
            val = np.nan
        else:
            try:
                val = mfunc(all_y_true_concat_val, all_y_pred_concat_val)
            except Exception:
                val = np.nan
        all_col_val.append(f"{val:.3f}" if not np.isnan(val) else "--")
    table_val["All"] = all_col_val
    cols_val = ["All"] + [c for c in table_val.columns if c != "All"]
    table_val = table_val[cols_val]
    # Save validation-only tables
    output_csv_val = os.path.splitext(args.output_csv)[0] + "_val.csv"
    output_latex_val = os.path.splitext(args.output_latex)[0] + "_val.tex"
    table_val.to_csv(output_csv_val)
    with open(output_latex_val, "w") as f:
        f.write(
            table_val.to_latex(
                escape=False,
                column_format='l' + 'c'*len(table_val.columns),
                bold_rows=True,
                caption="Evaluation metrics on validation set (overall, per task)",
                label="tab:raw_eval_metrics_val",
                longtable=False,
                multicolumn=True,
                multicolumn_format='c',
                na_rep='--',
                index=True,
                float_format=".3f",
                buf=None,
            )
        )
    print(f"Saved validation metrics to {output_csv_val} and {output_latex_val}")


if __name__ == "__main__":
    main()
