import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
)
from xgboost import XGBClassifier
import json
import warnings
from data.data_loader import extract_subject_id, extract_acq_id
import nibabel as nib
from classical.feat_extraction import NeonatalMRIQualityAssessment
from visualization.plot_results import visualize_results
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from psutil import cpu_count
import joblib

warnings.filterwarnings("ignore")

def process_subject(subject_data, bids_dir):
    """Process a single subject's data and extract features"""
    filename = subject_data["filename"]
    print(f"Processing {filename}")
    data_type = subject_data["data_type"]
    subject_id = extract_subject_id(os.path.basename(filename))
    orientation = extract_acq_id(os.path.basename(filename))

    # Extract run from the orientation
    acq_types = {
        "axi": 1,
        "cor": 2,
        "sag": 3
    }
    run = acq_types[orientation]

    img_path = filename
    if img_path is None:
        print(f"Could not find path for {filename}")
        return None

    mask_path = os.path.join(
        bids_dir,
        "derivatives",
        "masks",
        f"sub-{subject_id}",
        "ses-01",
        "anat",
        f"sub-{subject_id}_ses-01_acq-{orientation}_run-{run}_mask.nii.gz",
    )

    try:
        # Load mask and image
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(int)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

    print("Processing image: ", img_path)
    
    # Process image and extract features
    qa = NeonatalMRIQualityAssessment(img_path, mask_data)
    features = qa.extract_all_quality_markers()
    
    # Create features directory in the same location as the scan
    scan_dir = os.path.dirname(img_path)
    features_dir = os.path.join(scan_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Save features as JSON using the same filename as the scan
    features_file = os.path.join(
        features_dir,
        os.path.basename(img_path).replace('.nii.gz', '.json')
    )
    
    # Prepare data for JSON serialization
    features_data = {
        'features': features,
        'metadata': {
            'subject_id': subject_id,
            'data_type': data_type,
            'orientation': orientation,
            'run': run,
            'labels': {col: subject_data[col] for col in subject_data.keys() 
                      if col not in ["filename", "data_type"]}
        }
    }
    
    # Save to JSON
    with open(features_file, 'w') as f:
        json.dump(features_data, f, indent=2)
    
    # Store results for the combined dataset
    result = {
        'features': features,  # Store the full features dictionary
        'feature_names': list(features.keys()),  # Store feature names
        'subject_id': subject_id,
        'data_type': data_type,
        'labels': {col: subject_data[col] for col in subject_data.keys() 
                  if col not in ["filename", "data_type"]}
    }
    
    # Clear memory
    del mask, mask_data, qa, features
    import gc
    gc.collect()
    
    return result


def load_data(bids_dir, qc_csv_path, augmented_dir, augmented_qc_csv_path, batch_size=10):
    """Load data and create feature matrix for both raw and augmented data"""
    # Load raw data
    qc_df = pd.read_csv(qc_csv_path)
    qc_df["data_type"] = "raw"  # Add column to indicate raw data

    # Load augmented data
    aug_qc_df = pd.read_csv(augmented_qc_csv_path)
    aug_qc_df["data_type"] = "augmented"  # Add column to indicate augmented data

    # Combine dataframes
    combined_df = pd.concat([qc_df, aug_qc_df], ignore_index=True)
    
    # Remove rows with NaN filenames
    combined_df = combined_df.dropna(subset=['filename'])
    
    # Collect all features and metadata
    features_list = []
    # Only include specified QC label columns
    qc_columns = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
    labels_dict = {col: [] for col in qc_columns}
    subject_ids = []
    data_types = []
    feature_names = None
    
    # First, check which files need feature computation
    files_to_process = []
    for _, row in combined_df.iterrows():
        filename = row['filename']
        scan_dir = os.path.dirname(filename)
        features_dir = os.path.join(scan_dir, "features")
        features_file = os.path.join(
            features_dir,
            os.path.basename(filename).replace('.nii.gz', '.json')
        )
        
        if not os.path.exists(features_file):
            files_to_process.append(row.to_dict())
    
    # Process files in parallel if needed
    if files_to_process:
        print(f"\nComputing features for {len(files_to_process)} files...")
        n_cores = max(1, cpu_count() - 1)
        with Pool(n_cores) as pool:
            process_func = partial(process_subject, bids_dir=bids_dir)
            results = list(tqdm(
                pool.imap(process_func, files_to_process),
                total=len(files_to_process),
                desc="Processing subjects"
            ))
    
    # Now load all features (both computed and existing)
    print("\nLoading all features...")
    for _, row in combined_df.iterrows():
        filename = row['filename']
        scan_dir = os.path.dirname(filename)
        features_dir = os.path.join(scan_dir, "features")
        features_file = os.path.join(
            features_dir,
            os.path.basename(filename).replace('.nii.gz', '.json')
        )
        
        if not os.path.exists(features_file):
            print(f"Features file not found for {filename}, skipping...")
            continue
            
        # Load features from JSON
        with open(features_file, 'r') as f:
            features_data = json.load(f)
            
        # Get feature names from first file
        if feature_names is None:
            feature_names = list(features_data['features'].keys())
            
        # Extract features in the same order as feature_names
        features_list.append([features_data['features'][name] for name in feature_names])
        
        # Extract metadata
        subject_ids.append(features_data['metadata']['subject_id'])
        data_types.append(row['data_type'])
        for col in labels_dict.keys():
            labels_dict[col].append(row[col])
    
    if not features_list:
        raise ValueError("No valid features were obtained")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y_dict = {col: np.array(labels_dict[col]) for col in labels_dict.keys()}
    subject_ids = np.array(subject_ids)
    data_types = np.array(data_types)

    # Create DataFrame with features
    features_df = pd.DataFrame(X, columns=feature_names)
    
    # Add metadata columns
    features_df['subject_id'] = subject_ids
    features_df['data_type'] = data_types
    
    # Add only QC label columns
    for col in qc_columns:
        features_df[col] = y_dict[col]
    
    # Define columns to exclude from feature processing
    exclude_columns = [
        'subject_id', 'data_type', 'filename_old'
    ] + qc_columns  # Add QC columns to exclude list
    
    # Handle NaN and infinite values only in feature columns
    feature_columns = [col for col in features_df.columns if col not in exclude_columns]
    problematic_columns = features_df[feature_columns].columns[
        (features_df[feature_columns].isna().any()) | 
        (np.isinf(features_df[feature_columns].select_dtypes(include=np.number)).any())
    ].tolist()
    
    if problematic_columns:
        print("\nColumns with NaN or infinite values:")
        for col in problematic_columns:
            nan_count = features_df[col].isna().sum()
            inf_count = np.isinf(features_df[col]).sum() if features_df[col].dtype in [np.float64, np.float32] else 0
            print(f"  {col}: {nan_count} NaN values, {inf_count} infinite values")
        
        print(f"\nRemoving {len(problematic_columns)} columns with NaN or infinite values")
        features_df = features_df.drop(columns=problematic_columns)
        # Update feature_names to match remaining columns
        feature_names = [name for name in feature_names if name not in problematic_columns]
    
    # Update all arrays with cleaned data
    X = features_df[feature_names].values
    subject_ids = features_df['subject_id'].values
    data_types = features_df['data_type'].values
    for col in qc_columns:
        if col in features_df.columns:
            y_dict[col] = features_df[col].values
    
    # Normalize features using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save the scaler for use in validation/inference
    import joblib
    out_dir = 'results_classical_FINAL'
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, 'feature_scaler.pkl'))
    print(f"\nFeature scaler saved to {out_dir}/feature_scaler.pkl")

    # Save features to disk
    os.makedirs(out_dir, exist_ok=True)
    features_df.to_csv(os.path.join(out_dir, 'features.csv'), index=False)
    print(f"\nFeatures saved to {out_dir}/features.csv")

    return X, y_dict, subject_ids, data_types, feature_names


def train_and_evaluate(X, y_dict, subject_ids, data_types, feature_names, n_splits=10):
    """Train and evaluate models using subject-wise cross-validation"""
    # Initialize StratifiedGroupKFold for subject-wise splitting
    group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define models to evaluate
    models = {
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,  # For our 0,1,2 QC scores
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        ),
    }

    # Initialize results dictionary with separate metrics for all and raw data
    results = {
        model_name: {
            col: {
                "all": {
                    "accuracy": [],
                    "f1_micro": [],
                    "f1_macro": [],
                    "f1_weighted": [],
                    "f2_micro": [],
                    "f2_macro": [],
                    "f2_weighted": [],
                    "precision_micro": [],
                    "precision_macro": [],
                    "precision_weighted": [],
                    "recall_micro": [],
                    "recall_macro": [],
                    "recall_weighted": [],
                    "predictions": [],
                    "true_labels": [],
                    "feature_importance": [] if model_name in ["rf", "xgb"] else None,
                },
                "raw": {
                    "accuracy": [],
                    "f1_micro": [],
                    "f1_macro": [],
                    "f1_weighted": [],
                    "f2_micro": [],
                    "f2_macro": [],
                    "f2_weighted": [],
                    "precision_micro": [],
                    "precision_macro": [],
                    "precision_weighted": [],
                    "recall_micro": [],
                    "recall_macro": [],
                    "recall_weighted": [],
                    "predictions": [],
                    "true_labels": [],
                    "feature_importance": [] if model_name in ["rf", "xgb"] else None,
                },
            }
            for col in y_dict.keys()
        }
        for model_name in models.keys()
    }

    # Train and evaluate for each fold
    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(X, y_dict["Noise"], subject_ids)
    ):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nModel: {model_name}")

            # Train and evaluate for each QC category
            for col in y_dict.keys():
                y_train = y_dict[col][train_idx]
                y_val = y_dict[col][val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred_all = model.predict(X_val)

                # Calculate metrics for all data
                results[model_name][col]["all"]["accuracy"].append(
                    accuracy_score(y_val, y_pred_all)
                )
                results[model_name][col]["all"]["f1_micro"].append(
                    f1_score(y_val, y_pred_all, average="micro")
                )
                results[model_name][col]["all"]["f1_macro"].append(
                    f1_score(y_val, y_pred_all, average="macro")
                )
                results[model_name][col]["all"]["f1_weighted"].append(
                    f1_score(y_val, y_pred_all, average="weighted")
                )
                results[model_name][col]["all"]["f2_micro"].append(
                    fbeta_score(y_val, y_pred_all, beta=2, average="micro")
                )
                results[model_name][col]["all"]["f2_macro"].append(
                    fbeta_score(y_val, y_pred_all, beta=2, average="macro")
                )
                results[model_name][col]["all"]["f2_weighted"].append(
                    fbeta_score(y_val, y_pred_all, beta=2, average="weighted")
                )
                results[model_name][col]["all"]["precision_micro"].append(
                    precision_score(y_val, y_pred_all, average="micro")
                )
                results[model_name][col]["all"]["precision_macro"].append(
                    precision_score(y_val, y_pred_all, average="macro")
                )
                results[model_name][col]["all"]["precision_weighted"].append(
                    precision_score(y_val, y_pred_all, average="weighted")
                )
                results[model_name][col]["all"]["recall_micro"].append(
                    recall_score(y_val, y_pred_all, average="micro")
                )
                results[model_name][col]["all"]["recall_macro"].append(
                    recall_score(y_val, y_pred_all, average="macro")
                )
                results[model_name][col]["all"]["recall_weighted"].append(
                    recall_score(y_val, y_pred_all, average="weighted")
                )
                results[model_name][col]["all"]["predictions"].append(
                    y_pred_all.tolist()
                )
                results[model_name][col]["all"]["true_labels"].append(y_val.tolist())

                # Calculate metrics for raw data only
                raw_mask = data_types[val_idx] == "raw"
                y_val_raw = y_val[raw_mask]
                y_pred_raw = y_pred_all[raw_mask]

                results[model_name][col]["raw"]["accuracy"].append(
                    accuracy_score(y_val_raw, y_pred_raw)
                )
                results[model_name][col]["raw"]["f1_micro"].append(
                    f1_score(y_val_raw, y_pred_raw, average="micro")
                )
                results[model_name][col]["raw"]["f1_macro"].append(
                    f1_score(y_val_raw, y_pred_raw, average="macro")
                )
                results[model_name][col]["raw"]["f1_weighted"].append(
                    f1_score(y_val_raw, y_pred_raw, average="weighted")
                )
                results[model_name][col]["raw"]["f2_micro"].append(
                    fbeta_score(y_val_raw, y_pred_raw, beta=2, average="micro")
                )
                results[model_name][col]["raw"]["f2_macro"].append(
                    fbeta_score(y_val_raw, y_pred_raw, beta=2, average="macro")
                )
                results[model_name][col]["raw"]["f2_weighted"].append(
                    fbeta_score(y_val_raw, y_pred_raw, beta=2, average="weighted")
                )
                results[model_name][col]["raw"]["precision_micro"].append(
                    precision_score(y_val_raw, y_pred_raw, average="micro")
                )
                results[model_name][col]["raw"]["precision_macro"].append(
                    precision_score(y_val_raw, y_pred_raw, average="macro")
                )
                results[model_name][col]["raw"]["precision_weighted"].append(
                    precision_score(y_val_raw, y_pred_raw, average="weighted")
                )
                results[model_name][col]["raw"]["recall_micro"].append(
                    recall_score(y_val_raw, y_pred_raw, average="micro")
                )
                results[model_name][col]["raw"]["recall_macro"].append(
                    recall_score(y_val_raw, y_pred_raw, average="macro")
                )
                results[model_name][col]["raw"]["recall_weighted"].append(
                    recall_score(y_val_raw, y_pred_raw, average="weighted")
                )
                results[model_name][col]["raw"]["predictions"].append(
                    y_pred_raw.tolist()
                )
                results[model_name][col]["raw"]["true_labels"].append(
                    y_val_raw.tolist()
                )

                # Store feature importance for tree-based models
                if model_name in ["rf", "xgb"]:
                    results[model_name][col]["all"]["feature_importance"].append(
                        model.feature_importances_.tolist()
                    )
                    results[model_name][col]["raw"]["feature_importance"].append(
                        model.feature_importances_.tolist()
                    )

                # Clear memory
                del y_pred_all, y_pred_raw
                import gc
                gc.collect()

                print(f"{col}:")
                print("All data:")
                print(f"  Accuracy: {results[model_name][col]['all']['accuracy'][-1]:.3f}")
                print(f"  F1 (micro/macro/weighted): {results[model_name][col]['all']['f1_micro'][-1]:.3f}/{results[model_name][col]['all']['f1_macro'][-1]:.3f}/{results[model_name][col]['all']['f1_weighted'][-1]:.3f}")
                print(f"  F2 (micro/macro/weighted): {results[model_name][col]['all']['f2_micro'][-1]:.3f}/{results[model_name][col]['all']['f2_macro'][-1]:.3f}/{results[model_name][col]['all']['f2_weighted'][-1]:.3f}")
                print(f"  Precision (micro/macro/weighted): {results[model_name][col]['all']['precision_micro'][-1]:.3f}/{results[model_name][col]['all']['precision_macro'][-1]:.3f}/{results[model_name][col]['all']['precision_weighted'][-1]:.3f}")
                print(f"  Recall (micro/macro/weighted): {results[model_name][col]['all']['recall_micro'][-1]:.3f}/{results[model_name][col]['all']['recall_macro'][-1]:.3f}/{results[model_name][col]['all']['recall_weighted'][-1]:.3f}")
                print("Raw data only:")
                print(f"  Accuracy: {results[model_name][col]['raw']['accuracy'][-1]:.3f}")
                print(f"  F1 (micro/macro/weighted): {results[model_name][col]['raw']['f1_micro'][-1]:.3f}/{results[model_name][col]['raw']['f1_macro'][-1]:.3f}/{results[model_name][col]['raw']['f1_weighted'][-1]:.3f}")
                print(f"  F2 (micro/macro/weighted): {results[model_name][col]['raw']['f2_micro'][-1]:.3f}/{results[model_name][col]['raw']['f2_macro'][-1]:.3f}/{results[model_name][col]['raw']['f2_weighted'][-1]:.3f}")
                print(f"  Precision (micro/macro/weighted): {results[model_name][col]['raw']['precision_micro'][-1]:.3f}/{results[model_name][col]['raw']['precision_macro'][-1]:.3f}/{results[model_name][col]['raw']['precision_weighted'][-1]:.3f}")
                print(f"  Recall (micro/macro/weighted): {results[model_name][col]['raw']['recall_micro'][-1]:.3f}/{results[model_name][col]['raw']['recall_macro'][-1]:.3f}/{results[model_name][col]['raw']['recall_weighted'][-1]:.3f}")

            # Clear model memory
            del model
            gc.collect()

    # Calculate and print average results
    print("\nAverage Results:")
    for model_name in models.keys():
        print(f"\nModel: {model_name}")
        for col in y_dict.keys():
            print(f"\n{col}:")
            for data_type in ["all", "raw"]:
                print(f"\n{data_type.capitalize()} data:")
                metrics = [
                    "accuracy", "f1_micro", "f1_macro", "f1_weighted",
                    "f2_micro", "f2_macro", "f2_weighted",
                    "precision_micro", "precision_macro", "precision_weighted",
                    "recall_micro", "recall_macro", "recall_weighted"
                ]
                for metric in metrics:
                    mean_score = np.mean(results[model_name][col][data_type][metric])
                    std_score = np.std(results[model_name][col][data_type][metric])
                    print(f"  {metric}: {mean_score:.3f} ± {std_score:.3f}")

    # Save results for later visualization
    save_results(results)

    # Train and save full models for each task
    out_dir = 'results_classical_FINAL'
    os.makedirs(out_dir, exist_ok=True)
    print("\nTraining and saving full models on all data...")
    for col in y_dict.keys():
        print(f"  Training full RandomForest for {col}...")
        full_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        full_model.fit(X, y_dict[col])
        model_path = os.path.join(
            out_dir, f'{col}_rf.pkl'
        )
        joblib.dump(full_model, model_path)
        print(f"    Saved to {model_path}")
        # Save feature names used for this model
        feature_names_path = os.path.join(
            out_dir, f'{col}_rf_features.json'
        )
        with open(feature_names_path, 'w') as f:
            json.dump(list(feature_names), f, indent=2)
        print(f"    Feature names saved to {feature_names_path}")

        # Train and save XGBoost model for each task
        print(f"  Training full XGBoost for {col}...")
        full_xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,  # For our 0,1,2 QC scores
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        full_xgb_model.fit(X, y_dict[col])
        xgb_model_path = os.path.join(
            out_dir, f'{col}_xgb.pkl'
        )
        joblib.dump(full_xgb_model, xgb_model_path)
        print(f"    Saved to {xgb_model_path}")
        # Save feature names used for this model
        xgb_feature_names_path = os.path.join(
            out_dir, f'{col}_xgb_features.json'
        )
        with open(xgb_feature_names_path, 'w') as f:
            json.dump(list(feature_names), f, indent=2)
        print(f"    Feature names saved to {xgb_feature_names_path}")


def save_results(results):
    """Save results to JSON file for later visualization"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for model_name in results:
        serializable_results[model_name] = {}
        for col in results[model_name]:
            serializable_results[model_name][col] = {}
            for data_type in ["all", "raw"]:
                serializable_results[model_name][col][data_type] = {}
                for metric in results[model_name][col][data_type]:
                    if (
                        metric == "feature_importance"
                        and results[model_name][col][data_type][metric] is not None
                    ):
                        # Average feature importance across folds
                        avg_importance = np.mean(
                            results[model_name][col][data_type][metric], axis=0
                        )
                        serializable_results[model_name][col][data_type][
                            metric
                        ] = avg_importance.tolist()
                    elif metric in ["predictions", "true_labels"]:
                        # Flatten the lists of predictions and true labels
                        serializable_results[model_name][col][data_type][metric] = [
                            item
                            for sublist in results[model_name][col][data_type][metric]
                            for item in sublist
                        ]
                    elif results[model_name][col][data_type][metric] is not None:
                        # Calculate mean and std for each metric
                        mean_score = np.mean(results[model_name][col][data_type][metric])
                        std_score = np.std(results[model_name][col][data_type][metric])
                        serializable_results[model_name][col][data_type][metric] = {
                            "mean": float(mean_score),
                            "std": float(std_score)
                        }

    # Create results directory if it doesn't exist
    out_dir = 'results_classical_FINAL'
    os.makedirs(out_dir, exist_ok=True)

    # Save detailed results
    with open(os.path.join(out_dir, "qc_prediction_results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Create summary results for each model
    summary_results = {}
    for model_name in results:
        summary_results[model_name] = {}
        for col in results[model_name]:
            summary_results[model_name][col] = {}
            for data_type in ["all", "raw"]:
                summary_results[model_name][col][data_type] = {}
                metrics = [
                    "accuracy", "f1_micro", "f1_macro", "f1_weighted",
                    "f2_micro", "f2_macro", "f2_weighted",
                    "precision_micro", "precision_macro", "precision_weighted",
                    "recall_micro", "recall_macro", "recall_weighted"
                ]
                for metric in metrics:
                    mean_score = np.mean(results[model_name][col][data_type][metric])
                    std_score = np.std(results[model_name][col][data_type][metric])
                    summary_results[model_name][col][data_type][metric] = {
                        "mean": float(mean_score),
                        "std": float(std_score)
                    }

    # Calculate task-averaged metrics
    task_averaged = {}
    for model_name in results:
        task_averaged[model_name] = {}
        for data_type in ["all", "raw"]:
            task_averaged[model_name][data_type] = {}
            metrics = [
                "accuracy", "f1_micro", "f1_macro", "f1_weighted",
                "f2_micro", "f2_macro", "f2_weighted",
                "precision_micro", "precision_macro", "precision_weighted",
                "recall_micro", "recall_macro", "recall_weighted"
            ]
            for metric in metrics:
                # Collect all task means for this metric
                task_means = []
                task_stds = []
                for col in results[model_name]:
                    mean_score = np.mean(results[model_name][col][data_type][metric])
                    std_score = np.std(results[model_name][col][data_type][metric])
                    task_means.append(mean_score)
                    task_stds.append(std_score)
                
                # Calculate average across tasks
                avg_mean = np.mean(task_means)
                avg_std = np.mean(task_stds)  # Average of task-specific stds
                std_of_means = np.std(task_means)  # Std of task means
                
                task_averaged[model_name][data_type][metric] = {
                    "mean": float(avg_mean),
                    "std_of_means": float(std_of_means),
                    "avg_std": float(avg_std)
                }

    # Add task-averaged metrics to summary results
    summary_results["task_averaged"] = task_averaged

    # Save summary results
    with open(os.path.join(out_dir, "qc_prediction_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)

    # --- Export CSV and LaTeX tables for each model's raw data ---
    for model_name in results:
        if model_name == "task_averaged":
            continue
        # Get all tasks and metrics
        tasks = list(summary_results[model_name].keys())
        metrics = [
            "accuracy", "f1_micro", "f1_macro", "f1_weighted",
            "f2_micro", "f2_macro", "f2_weighted",
            "precision_micro", "precision_macro", "precision_weighted",
            "recall_micro", "recall_macro", "recall_weighted"
        ]
        # Build table: rows=metrics, columns=tasks, cell="mean ± std"
        table = pd.DataFrame(index=metrics, columns=tasks)
        for task in tasks:
            for metric in metrics:
                m = summary_results[model_name][task]["raw"][metric]["mean"]
                s = summary_results[model_name][task]["raw"][metric]["std"]
                table.at[metric, task] = f"{m:.3f} ± {s:.3f}"
        # Save as CSV
        csv_path = os.path.join(out_dir, f"raw_results_{model_name}.csv")
        table.to_csv(csv_path)
        # Save as LaTeX (booktabs)
        latex_path = os.path.join(out_dir, f"raw_results_{model_name}.tex")
        caption = f"Cross-validation results (mean $\\pm$ std) for model {model_name} on raw data."
        label = f"tab:raw_results_{model_name}"
        with open(latex_path, "w") as f:
            f.write(table.to_latex(
                escape=False,
                column_format='l' + 'c'*len(tasks),
                bold_rows=True,
                caption=caption,
                label=label,
                longtable=False,
                multicolumn=True,
                multicolumn_format='c',
                na_rep='--',
                index=True,
                float_format=".3f",
                buf=None,
            ))

    # Print task-averaged results
    print("\nTask-Averaged Results:")
    for model_name in task_averaged:
        print(f"\nModel: {model_name}")
        for data_type in ["all", "raw"]:
            print(f"\n{data_type.capitalize()} data:")
            for metric in task_averaged[model_name][data_type]:
                mean = task_averaged[model_name][data_type][metric]["mean"]
                std_of_means = task_averaged[model_name][data_type][metric]["std_of_means"]
                print(f"  {metric}: {mean:.3f} ± {std_of_means:.3f}")

    print("\nResults saved to:")
    print(f"- {out_dir}/qc_prediction_results.json (detailed results)")
    print(f"- {out_dir}/qc_prediction_summary.json (summary results)")


def main():
    # Set paths
    bids_dir = "data/LISA2025/BIDS_norm"
    qc_csv_path = "data/LISA2025/BIDS_norm/LISA_2025_bids.csv"
    augmented_dir = "data/LISA2025/BIDS_norm/derivatives/augmented_FINAL_Distortion"
    augmented_qc_csv_path = "data/LISA2025/BIDS_norm/derivatives/augmented_FINAL_Distortion/augmented_labels.csv"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y_dict, subject_ids, data_types, feature_names = load_data(
        bids_dir, qc_csv_path, augmented_dir, augmented_qc_csv_path
    )

    # Train and evaluate models
    print("\nTraining and evaluating models...")
    train_and_evaluate(X, y_dict, subject_ids, data_types, feature_names)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(feature_names)


if __name__ == "__main__":
    main()
