# LISA 2025 - Task 01: Quality Control for Brain MRI

Complete implementation for the MICCAI 2025 LISA Challenge Task 01 - Quality Control of brain MRI scans with **five different approaches** for ordinal classification across 7 artifact types.

## Overview

The task involves classifying brain MRI scans into 3 ordinal quality levels (0: Good, 1: Moderate, 2: Bad) for each of 7 artifact types:
- Noise
- Zipper
- Positioning
- Banding
- Motion
- Contrast
- Distortion

## Five Model Approaches

### 1. Simple DenseNet (src/models/train.py)
Basic deep learning approach using DenseNet264 with CrossEntropy loss.

**Features**:
- DenseNet264 architecture (3D medical imaging)
- CrossEntropy loss
- **StratifiedGroupKFold** for subject-level splitting
- Early stopping with patience
- Dynamic class balancing (10% of class 0)

**Usage**:
```bash
cd src/models
python train.py
```

### 2. Ordinal Regression Model (src/models/train_ordinal.py)
Advanced ordinal classification using Cumulative Link Model (CLM) with QWK loss.

**Features**:
- OrdinalDenseNet with CLM output layer
- **QWK loss** + optional **Focal loss** for ordinal relationships
- **Class weighting** for imbalanced data
- Ordinal-specific metrics (accuracy off-by-1, MAE)
- WandB integration for experiment tracking
- Smaller spatial size (150³) for memory efficiency

**Usage**:
```bash
cd src/models
python train_ordinal.py
```

### 3. Classical Machine Learning (src/models/train_classical.py)
Feature-based approach with traditional ML classifiers.

**Features**:
- Three classifiers: **Random Forest**, **SVM**, **XGBoost**
- Custom MRI quality feature extraction
- **StratifiedGroupKFold** cross-validation (10 folds)
- Feature standardization with saved scaler
- Exports results as CSV and LaTeX tables
- Saves trained models for inference

**Usage**:
```bash
cd src/models
python train_classical.py
```

### 4. Baseline with Task-Specific Augmentation (baseline/)
Enhanced DenseNet with intelligent task-specific data augmentation.

**Files**:
- `baseline_lisa2025.py` - Main training with offline/online augmentation
- `baseline_eval_valsplit.py` - Evaluation on validation split
- `baseline_validation.py` - Inference on new data

**Key Innovation - Task-Specific Augmentations**:
- **Noise task**: Only spatial augmentations (avoid adding noise)
- **Zipper task**: Spatial + mild intensity
- **Positioning task**: Only intensity (NO spatial transforms)
- **Banding task**: Only spatial (avoid intensity changes)
- **Motion task**: Only intensity (NO spatial transforms)
- **Contrast task**: Only spatial (NO contrast modifications)
- **Distortion task**: Only intensity (NO spatial distortions)

**Usage**:
```bash
cd baseline
python baseline_lisa2025.py
```

### 5. Supervised Contrastive Learning with Ensemble (SupCon/) ⭐

**State-of-the-art approach** combining contrastive learning, ordinal awareness, and model ensembling.

**Core Files**:
- `train_contrastive.py` - **Two-stage contrastive pre-training**
- `train.py`, `train_emd.py`, `train_focal_raw.py` - Multiple training strategies
- `ensemble_eval.py` - **Ensemble prediction** combining multiple models
- `losses.py` - Custom losses (Ordinal Contrastive, QWK, EMD, Focal)
- `model.py` - Model architectures
- `validation.py` - Inference script for deployment
- `approach.md` - Detailed methodology documentation

## Directory Structure

```
Task01/
├── README.md (this file)
├── requirements.txt
├── config/
│   └── default.yaml
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── augment.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py (Simple DenseNet)
│   │   ├── train_ordinal.py (Ordinal regression)
│   │   └── train_classical.py (Classical ML)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── ordinal.py (CLM, QWK loss, Focal loss)
│   │   ├── data.py
│   │   ├── checkpoint.py
│   │   ├── seed.py
│   │   └── training.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── dataset.py
│   ├── classical/
│   │   └── feat_extraction.py
│   └── visualization/
│       └── plot_results.py
├── baseline/
│   ├── baseline_lisa2025.py
│   ├── baseline_eval_valsplit.py
│   └── baseline_validation.py
└── SupCon/
    ├── approach.md (detailed methodology)
    ├── README_validation.md
    ├── train_contrastive.py
    ├── train.py
    ├── train_emd.py
    ├── train_emd_raw.py
    ├── train_focal_raw.py
    ├── ensemble_eval.py ⭐
    ├── losses.py
    ├── model.py
    ├── validation.py
    ├── validation_bayesian.py
    ├── validation_bayesian_val.py
    ├── bayesian_param_search.py
    ├── test.py
    └── test_augment.py
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your data is organized in BIDS format:
```
data/LISA2025/BIDS/
├── sub-001/
│   └── ses-01/
│       └── anat/
│           ├── sub-001_ses-01_acq-axi_run-1_T1w.nii.gz
│           ├── sub-001_ses-01_acq-cor_run-2_T1w.nii.gz
│           └── sub-001_ses-01_acq-sag_run-3_T1w.nii.gz
├── LISA_2025_bids.csv
└── derivatives/
    ├── masks/ (for classical ML)
    └── augmented/ (generated by augmentation script)
```

### 3. Generate Augmented Data

```bash
cd src/preprocessing
python augment.py
```

Creates synthetic artifacts with controlled severity in `data/LISA2025/BIDS/derivatives/augmented/`

## Model Comparison

| Approach | Architecture | Loss Function | Training Strategy | Best For |
|----------|--------------|---------------|-------------------|----------|
| **Simple DenseNet** | DenseNet264 | CrossEntropy | Class subsampling | Baseline performance |
| **Ordinal Regression** | DenseNet264 + CLM | QWK + Focal | Class weighting | Ordinal tasks |
| **Classical ML** | RF/SVM/XGBoost | Default | Feature extraction + CV | Fast experimentation, interpretability |
| **Baseline** | DenseNet264 | CrossEntropy | Task-specific augmentation | Artifact-specific learning |
| **SupCon + Ensemble** | DenseNet264 + Projection | Ordinal Contrastive + QWK | Two-stage + Ensemble | **Best overall performance** |

## Evaluation

### Deep Learning Models (Approaches 1, 2, 4, 5)

```bash
cd src/evaluation
python evaluate.py --task Noise --results_dir results_dense264
python evaluate.py  # Evaluates all tasks
```

### Classical ML (Approach 3)

Results are automatically saved during training:
- `results_classical_FINAL/qc_prediction_summary.json`
- `results_classical_FINAL/raw_results_rf.csv`
- `results_classical_FINAL/raw_results_xgb.csv`

### Ensemble Evaluation (Approach 5)

```bash
cd SupCon
python ensemble_eval.py --model_dir results_ordinal_contrastive --output_csv ensemble_results.csv
```
