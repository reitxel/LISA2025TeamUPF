# Ordinal Contrastive Loss for Imbalanced LISA 2025 QC Challenge

## Overview

This implementation combines insights from two key papers to address the ordinal classification problem with imbalanced classes (0, 1, 2) in the LISA 2025 QC challenge:

1. **Supervised Contrastive Ordinal Loss (SCOL)** - Incorporates ordinal distance metrics into contrastive learning
2. **A Tale of Two Classes** - Addresses the collapse of contrastive representations in binary imbalanced settings

## Key Innovations

### 1. Ordinal Distance Weighting

Unlike standard supervised contrastive loss, our approach incorporates ordinal relationships:

```python
# Ordinal distance matrix for classes 0, 1, 2
distance(0,1) = 1/4  # Adjacent classes
distance(0,2) = 1    # Far classes
distance(1,2) = 1/4  # Adjacent classes
```

This ensures that confusing class 0 with class 2 incurs a higher penalty than confusing class 0 with class 1.

### 2. Imbalance Handling Strategies

#### Adaptive Class Weighting
- Tracks class frequencies during training
- Applies inverse frequency weights to give more importance to minority classes (1 and 2)

#### Prototype Alignment
- Initializes fixed prototypes for each class on the unit hypersphere
- Arranges prototypes to reflect ordinal relationships (classes 0, 1, 2 form a sequence)
- Only attracts samples to prototypes when similarity is below threshold (prevents over-clustering)

### 3. Two-Stage Training

1. **Contrastive Pre-training (50 epochs)**
   - Uses the ImbalancedOrdinalContrastiveLoss
   - Learns robust feature representations that respect both ordinal relationships and class imbalance
   
2. **Fine-tuning (50 epochs)**
   - Switches to classification head
   - Uses Quadratic Weighted Kappa (QWK) loss for ordinal classification
   - Lower learning rate for stable convergence

## Loss Function Details

### Contrastive Loss Component

```python
L_contrastive = -log(exp(sim(xi, xj)/τ) / Σ_k exp(sim(xi, xk) * w_ordinal(yi, yk)/τ))
```

Where:
- `sim(xi, xj)` is cosine similarity between embeddings
- `w_ordinal(yi, yk)` is the ordinal weight based on class distance
- `τ` is temperature parameter (0.07)

### Prototype Loss Component

```python
L_prototype = -log(exp(sim(xi, p_yi)/τ) / Σ_j exp(sim(xi, p_j)/τ))
```

Applied only when `sim(xi, p_yi) < 0.5` to prevent over-clustering.

### Total Loss

```python
L_total = L_contrastive + 0.1 * L_prototype
```

## Implementation Highlights

### DenseNet Architecture
- Uses DenseNet264 backbone (proven effective in medical imaging)
- Projection head for contrastive learning (1000 → 512 → 128)
- Classification head for fine-tuning (1000 → 512 → 3)

### Data Augmentation
- Spatial augmentations: random flips, rotations, affine transformations
- Ensures diverse positive pairs for contrastive learning

### Evaluation Metrics
- **Quadratic Weighted Kappa (QWK)**: Primary metric for ordinal classification
- **Off-by-one Accuracy**: Measures predictions within one class of ground truth
- **Per-class Accuracy**: Monitors performance on imbalanced classes
- **Embedding Quality Metrics**: Intra/inter-class distances, distance ratio

## Advantages Over Standard Approaches

1. **Prevents Representation Collapse**: Unlike standard SupCon which fails on imbalanced data, our approach maintains distinct class clusters

2. **Respects Ordinal Nature**: The ordinal distance weighting ensures the model learns that some misclassifications are worse than others

3. **Handles Extreme Imbalance**: The combination of adaptive weighting and prototype alignment ensures minority classes (1 and 2) are well-represented

4. **Two-Stage Benefits**: 
   - Contrastive pre-training learns robust features without overfitting to imbalanced labels
   - Fine-tuning with QWK loss optimizes directly for the ordinal classification task

## Usage

```python
# Initialize model
model = OrdinalContrastiveLightningModule(
    num_classes=3,
    temperature=0.07,
    use_prototypes=True,
    prototype_similarity_threshold=0.5,
    minority_threshold=0.2,
    class_weights=class_weights
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
```

## Expected Performance Improvements

Based on the papers' findings:
- **SCOL**: Up to 10% improvement in ordinal classification tasks
- **Tale of Two Classes**: Up to 35% improvement on binary imbalanced datasets

For the LISA 2025 QC challenge with ordinal imbalanced data, we expect:
- Significant improvement in minority class (1, 2) detection
- Better ordinal consistency (fewer large errors like predicting 0 as 2)
- More robust features that generalize across different scanners/sites