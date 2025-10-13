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
from monai.networks.nets import DenseNet264
from monai.data import DataLoader, Dataset

def get_inference_transforms(spatial_size=(150, 150, 150)):
    return Compose([
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
        ToTensord(keys=["img"], dtype=torch.float32),
    ])

def prepare_valsplit(task, csv_path, random_state=42):
    df = pd.read_csv(csv_path)
    df = df[df["filename"].notna()]
    mask = df[task].isin([0, 1, 2])
    df = df[mask]
    filenames = df["filename"].astype(str).values
    labels = df[task].astype(int).values
    subjects = [os.path.basename(f).split('_')[0] for f in filenames]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf.split(filenames, labels, subjects))
    val_files = [{"img": filenames[i], "label": labels[i]} for i in val_idx]
    return val_files

def load_model(model_path, device="cuda"):
    model = DenseNet264(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    parser = argparse.ArgumentParser(description="Evaluate baseline models on validation split and output metrics table.")
    parser.add_argument("--model_dir", type=str, default="baseline/results_no_aug", help="Directory with trained models")
    parser.add_argument("--csv", type=str, default="data/LISA2025/BIDS/LISA_2025_bids.csv", help="CSV with image labels")
    parser.add_argument("--output_csv", type=str, default="baseline/results_no_aug/baseline_eval_valsplit_metrics.csv", help="Output CSV path")
    parser.add_argument("--output_latex", type=str, default="baseline/results_no_aug/baseline_eval_valsplit_metrics.tex", help="Output LaTeX path")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[150, 150, 150], help="Spatial size for image processing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for validation split")
    args = parser.parse_args()

    tasks = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
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
    transforms = get_inference_transforms(tuple(args.spatial_size))
    results = {task: {"y_true": [], "y_pred": []} for task in tasks}
    models = {}
    for task in tasks:
        model_path = os.path.join(args.model_dir, f"best_metric_model_LISA_LF_{task}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {task} at {model_path}. Skipping.")
            models[task] = None
            continue
        models[task] = load_model(model_path, args.device)
    for task in tasks:
        model = models[task]
        if model is None:
            continue
        val_files = prepare_valsplit(task, args.csv, args.random_state)
        y_true = [f["label"] for f in val_files]
        y_pred = []
        for f in val_files:
            try:
                pred = predict(model, f["img"], transforms, args.device)
            except Exception as e:
                print(f"Error processing {f['img']} for {task}: {e}")
                pred = -1
            y_pred.append(pred)
        valid = np.array(y_pred) != -1
        results[task]["y_true"] = np.array(y_true)[valid]
        results[task]["y_pred"] = np.array(y_pred)[valid]
    # Compute metrics
    table = pd.DataFrame(index=[m[0] for m in metrics], columns=tasks)
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
    cols = ["All"] + [c for c in table.columns if c != "All"]
    table = table[cols]
    table.to_csv(args.output_csv)
    # LaTeX output
    try:
        with open(args.output_latex, "w") as f:
            f.write(
                table.to_latex(
                    escape=False,
                    column_format='l' + 'c'*len(table.columns),
                    bold_rows=True,
                    caption="Evaluation metrics on validation split (overall, per task)",
                    label="tab:baseline_eval_valsplit_metrics",
                    longtable=False,
                    multicolumn=True,
                    multicolumn_format='c',
                    na_rep='--',
                    index=True,
                    float_format=".3f",
                    buf=None,
                )
            )
    except Exception as e:
        print(f"Could not write LaTeX file: {e}")
    print(f"Saved metrics to {args.output_csv} and {args.output_latex}")

if __name__ == "__main__":
    main() 