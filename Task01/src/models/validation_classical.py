"""
Validation script for classical ML models on LISA 2025 QC data.
This script loads trained classical models, extracts features, runs inference,
and saves results to a CSV file.
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import joblib
from classical.feat_extraction import NeonatalMRIQualityAssessment
import subprocess
from pathlib import Path
import json

# Define the 7 main QC tasks
TASKS = [
    "Noise", "Zipper", "Positioning", "Banding",
    "Motion", "Contrast", "Distortion"
]


def get_mask_path(mask_dir, image_filename):
    """
    Given the mask directory and image filename, construct the mask path.
    For LISA_VALIDATION_XXXX_LF_{axi,cor,sag}.nii.gz, mask is
    LISA_VALIDATION_XXXX_LF_{axi,cor,sag}_mask.nii.gz in mask_dir.
    Returns None if not a .nii.gz file.
    """
    basename = os.path.basename(image_filename)
    if not basename.endswith('.nii.gz'):
        print(f"Warning: Skipping non-NIfTI file: {basename}")
        return None
    mask_name = basename[:-7] + '_mask.nii.gz'
    mask_path = os.path.join(mask_dir, mask_name)
    return mask_path


def extract_features(image_path, mask_path):
    """
    Extract features from a single image and mask using
    NeonatalMRIQualityAssessment.
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


def run_bet(input_file: str, output_file: str) -> bool:
    """
    Run FSL's BET tool to create brain mask.
    Args:
        input_file (str): Path to input NIfTI file
        output_file (str): Path to output mask file (without _mask.nii.gz)
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cmd = ["bet", input_file, output_file, "-m", "-n", "-f", "0.2"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running BET on {input_file}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running BET on {input_file}: {str(e)}")
        return False


def run_model_type(model_type, args, features_df, feature_columns, task_feature_lists, filenames):
    """
    Load models of the given type (rf or xgb), predict, and save results to CSV.
    """
    models = {}
    feature_orders = {}
    for task in TASKS:
        model_path = os.path.join(
            args.model_dir, f"{task}_{model_type}.pkl"
        )
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for {task} ({model_type}) not found at "
                f"{model_path}. Skipping."
            )
            continue
        models[task] = joblib.load(model_path)
        if hasattr(models[task], "feature_names_in_"):
            feature_orders[task] = models[task].feature_names_in_
        else:
            feature_orders[task] = None
    if not models:
        print(
            f"No valid {model_type} models found in the model directory. "
            "Skipping."
        )
        return

    results = []
    for idx, filename in enumerate(filenames):
        result = {"filename": filename}
        if features_df.empty or filename not in features_df.index:
            for task in TASKS:
                result[task] = None
            results.append(result)
            continue
        for task in TASKS:
            if task in models:
                try:
                    feature_list = task_feature_lists.get(task)
                    if feature_list is not None:
                        features = features_df.loc[
                            filename, feature_list
                        ].to_dict()
                    else:
                        features = features_df.loc[
                            filename, feature_columns
                        ].to_dict()
                    feature_order = feature_orders[task]
                    if feature_order is not None:
                        feature_vec = [
                            features.get(f, np.nan)
                            for f in feature_order
                        ]
                    else:
                        feature_vec = [
                            features[k]
                            for k in sorted(features.keys())
                        ]
                    pred = models[task].predict([
                        feature_vec
                    ])[0]
                    result[task] = int(pred)
                except Exception as e:
                    print("Error predicting")
                    print(task)
                    print("for:")
                    print(filename)
                    print(str(e))
                    result[task] = None
            else:
                result[task] = None
        results.append(result)

    df = pd.DataFrame(results)
    for task in TASKS:
        if task not in df.columns:
            df[task] = None
    df = df[["filename"] + TASKS]
    df = df.sort_values(by="filename")
    output_csv = args.output_csv
    if output_csv.endswith(".csv"):
        output_csv = output_csv[:-4]
    output_csv = (
        f"{output_csv}_{model_type}.csv"
    )
    df.to_csv(output_csv, index=False)
    print(f"Results for {model_type} saved to {output_csv}")
    print(
        f"Processed {len(df)} images for {model_type}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Classical model validation for LISA 2025 QC."
    )
    parser.add_argument(
        "--input_dir", type=str, required=False,
        default="data/LISA2025/Task_1_Validation/",
        help=(
            "Directory with .nii.gz images "
            "(default: data/LISA2025/Task_1_Validation/)"
        )
    )
    parser.add_argument(
        "--mask_dir", type=str, required=False,
        default="data/LISA2025/Task_1_Validation_masks/",
        help=(
            "Directory with masks (sub-*/anat/*.nii.gz). "
            "If not provided, masks will be created next to input_dir. "
            "(default: None)"
        )
    )
    parser.add_argument(
        "--model_dir", type=str, required=False,
        default="results_paper/",
        help="Directory with pickled models (e.g., Noise_rf.pkl) "
             "(default: models)"
    )
    parser.add_argument(
        "--output_csv", type=str, required=False,
        default="results_paper/LISA_LF_QC_predictions.csv",
        help=(
            "Output CSV file path (suffixes _rf and _xgb will be added) "
            "(default: output.csv)"
        )
    )
    args = parser.parse_args()

    # Determine mask_dir
    if args.mask_dir is None:
        input_dir_path = Path(args.input_dir).resolve()
        parent_dir = input_dir_path.parent
        mask_dir = parent_dir / (input_dir_path.name + "_masks")
        args.mask_dir = str(mask_dir)
        print(
            "No mask directory provided. "
            "Masks will be created using BET in: "
            + str(args.mask_dir)
        )
        create_masks = True
    elif not os.path.exists(args.mask_dir):
        print(
            f"Mask directory {args.mask_dir} not found. "
            "Creating masks using BET..."
        )
        create_masks = True
    else:
        create_masks = False

    if create_masks:
        image_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
        os.makedirs(args.mask_dir, exist_ok=True)
        for image_path in image_files:
            mask_path = get_mask_path(args.mask_dir, image_path)
            if mask_path is None:
                continue
            if os.path.exists(mask_path):
                print(
                    f"Mask already exists for {image_path}, skipping."
                )
                continue
            print(
                f"Creating mask for {image_path} -> {mask_path}"
            )
            bet_output_prefix = mask_path[:-12]  # strip _mask.nii.gz
            success = run_bet(image_path, bet_output_prefix)
            if not success:
                print(
                    f"Failed to create mask for {image_path}"
                )
        print(
            "Mask creation complete. Using "
            + str(args.mask_dir)
            + " as mask directory."
        )

    # Find all images
    image_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    if not image_files:
        raise ValueError(f"No .nii.gz files found in {args.input_dir}")
    print(
        "Found " + str(len(image_files)) + " images to process."
    )

    # Process each image and collect features
    features_list = []
    filenames = []
    for i, image_path in enumerate(image_files):
        print(
            f"Processing image {i+1}/{len(image_files)}: "
            f"{os.path.basename(image_path)}"
        )
        mask_path = get_mask_path(args.mask_dir, image_path)
        if mask_path is None:
            print(f"Skipping {image_path}: not a NIfTI file.")
            continue
        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_path}, skipping.")
            continue
        features = extract_features(image_path, mask_path)
        if features is None:
            continue
        features_list.append(features)
        filenames.append(os.path.basename(image_path))

    # Build DataFrame and drop columns with NaN or infinite values (match train_classical.py)
    if features_list:
        features_df = pd.DataFrame(
            features_list,
            index=filenames
        )
        exclude_columns = [
            'subject_id', 'data_type', 'filename'
        ] + TASKS
        feature_columns = [
            col for col in features_df.columns
            if col not in exclude_columns
        ]
        is_inf = np.isinf(
            features_df[feature_columns].select_dtypes(
                include=[np.number]
            )
        )
        nan_mask = features_df[feature_columns].isna().any()
        inf_mask = is_inf.any()
        problematic_columns = features_df[feature_columns].columns[
            nan_mask | inf_mask
        ]
        if len(problematic_columns) > 0:
            print(
                "Dropping "
                + str(len(problematic_columns))
                + " columns with NaN or inf values:"
            )
            print(
                list(problematic_columns)
            )
            features_df = features_df.drop(columns=problematic_columns)
            feature_columns = [
                col for col in feature_columns
                if col not in problematic_columns
            ]
        # Load feature lists for each task from JSON (if available)
        task_feature_lists = {}
        for task in TASKS:
            feature_json = os.path.join(
                args.model_dir,
                f"{task}_rf_features.json"
            )
            if os.path.exists(feature_json):
                with open(feature_json, 'r') as f:
                    task_feature_lists[task] = json.load(f)
            else:
                task_feature_lists[task] = None
        # --- Normalize features using the scaler from training ---
        scaler_path = os.path.join(args.model_dir, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            features_df[feature_columns] = scaler.transform(
                features_df[feature_columns]
            )
            print(
                f"Applied feature normalization using scaler from {scaler_path}"
            )
        else:
            print(
                f"Warning: Scaler not found at {scaler_path}. "
                "Features will not be normalized."
            )
    else:
        features_df = pd.DataFrame()
        feature_columns = []
        task_feature_lists = {task: None for task in TASKS}

    print("feature_columns:")
    print(feature_columns)

    # Run both model types
    for model_type in ["rf", "xgb"]:
        run_model_type(model_type, args, features_df, feature_columns, task_feature_lists, filenames)


if __name__ == "__main__":
    main() 