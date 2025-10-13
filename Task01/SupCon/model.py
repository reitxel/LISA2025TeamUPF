import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from typing import Dict, Optional
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import the required losses from the previous artifact
from losses import (
    ImbalancedOrdinalContrastiveLoss, 
    OrdinalContrastiveDenseNet,
    OrdinalDenseNet,
    FocalLoss,
    EMDLoss
)

class OrdinalContrastiveLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training OrdinalContrastiveDenseNet with progressive 
    ordinal contrastive learning strategy.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        spatial_dims: int = 3,
        in_channels: int = 1,
        projection_dim: int = 64,
        ordinal_embedding_dim: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 180,
        # Contrastive loss parameters
        contrastive_strategy: str = "combined",  # "supervised_minority", "supervised_prototypes", "combined"
        temperature: float = 0.07,
        minority_threshold: float = 0.25,
        ordinal_weight_scale: float = 2.0,
        # Loss weights
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.5,
        ordinal_weight: float = 0.3,
        ordinal_contrastive_weight: float = 0.8,
        # Training phases
        warmup_epochs: int = 50,
        joint_epochs: int = 100,
        finetune_epochs: int = 30,
        # Model parameters
        dropout_prob: float = 0.3,
        use_ensemble_prediction: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = OrdinalContrastiveDenseNet(
            num_classes=num_classes,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            projection_dim=projection_dim,
            ordinal_embedding_dim=ordinal_embedding_dim,
            use_ordinal_head=True,
            dropout_prob=dropout_prob
        )
        
        # Contrastive loss
        self.contrastive_loss = ImbalancedOrdinalContrastiveLoss(
            num_classes=num_classes,
            strategy=contrastive_strategy,
            temperature=temperature,
            minority_threshold=minority_threshold,
            ordinal_weight_scale=ordinal_weight_scale,
            device=self.device
        )
        
        # Classification losses
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.ordinal_criterion = self._ordinal_loss
        
        # Training phase tracking
        self.warmup_epochs = warmup_epochs
        self.joint_epochs = joint_epochs
        self.finetune_epochs = finetune_epochs
        
    def _ordinal_loss(self, cumulative_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Ordinal classification loss using cumulative link model."""
        batch_size = cumulative_logits.shape[0]
        num_classes = cumulative_logits.shape[1] + 1
        
        # Create ordinal targets
        ordinal_targets = torch.zeros(batch_size, num_classes - 1, device=cumulative_logits.device)
        
        for i in range(batch_size):
            label = labels[i].item()
            if label > 0:
                ordinal_targets[i, :label] = 1
        
        # Binary cross-entropy for each cumulative probability
        loss = F.binary_cross_entropy_with_logits(cumulative_logits, ordinal_targets)
        return loss
    
    def _get_training_phase(self) -> str:
        """Determine current training phase based on epoch."""
        current_epoch = self.current_epoch
        
        if current_epoch < self.warmup_epochs:
            return 'warmup'
        elif current_epoch < self.warmup_epochs + self.joint_epochs:
            return 'joint'
        else:
            return 'finetune'
    
    def _compute_losses(self, outputs, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all losses based on current training phase."""
        training_phase = self._get_training_phase()
        losses = {}
        total_loss = 0.0
        
        # Handle both tensor and dictionary outputs
        if isinstance(outputs, torch.Tensor):
            # During warmup phase, outputs is just the projections tensor
            contrastive_loss = self.contrastive_loss(outputs, labels)
            losses['contrastive'] = contrastive_loss
            total_loss += self.hparams.contrastive_weight * contrastive_loss
        else:
            # Dictionary outputs (joint/finetune phases)
            # Contrastive loss (always included)
            if isinstance(outputs, dict) and 'projections' in outputs:
                contrastive_loss = self.contrastive_loss(outputs['projections'], labels)
                losses['contrastive'] = contrastive_loss
                total_loss += self.hparams.contrastive_weight * contrastive_loss
            
            # Progressive loss inclusion
            if training_phase in ['joint', 'finetune']:
                # Classification loss
                if isinstance(outputs, dict) and 'logits' in outputs:
                    classification_loss = self.classification_criterion(outputs['logits'], labels)
                    losses['classification'] = classification_loss
                    weight = (self.hparams.classification_weight * 
                             (2.0 if training_phase == 'finetune' else 1.0))
                    total_loss += weight * classification_loss
                
                # Ordinal contrastive loss
                if isinstance(outputs, dict) and 'ordinal_embeddings' in outputs:
                    ordinal_contrastive_loss = self.contrastive_loss(outputs['ordinal_embeddings'], labels)
                    losses['ordinal_contrastive'] = ordinal_contrastive_loss
                    total_loss += self.hparams.ordinal_contrastive_weight * ordinal_contrastive_loss
                
                # Ordinal classification loss
                if isinstance(outputs, dict) and 'cumulative_logits' in outputs:
                    ordinal_loss = self.ordinal_criterion(outputs['cumulative_logits'], labels)
                    losses['ordinal'] = ordinal_loss
                    weight = (self.hparams.ordinal_weight * 
                             (1.5 if training_phase == 'finetune' else 1.0))
                    total_loss += weight * ordinal_loss
        
        losses['total'] = total_loss
        return losses
    
    def _predict(self, outputs) -> torch.Tensor:
        """Make predictions using ensemble or single head."""
        if isinstance(outputs, torch.Tensor):
            # During warmup phase, no classification outputs available
            predictions = torch.zeros(outputs.shape[0], dtype=torch.long, device=self.device)
        else:
            # Dictionary outputs
            if (isinstance(outputs, dict) and 
                self.hparams.use_ensemble_prediction and 
                'ordinal_probs' in outputs and 'logits' in outputs):
                # Ensemble prediction
                standard_probs = torch.softmax(outputs['logits'], dim=1)
                ordinal_probs = outputs['ordinal_probs']
                ensemble_probs = 0.4 * standard_probs + 0.6 * ordinal_probs
                predictions = torch.argmax(ensemble_probs, dim=1)
            elif isinstance(outputs, dict) and 'logits' in outputs:
                # Standard prediction
                predictions = torch.argmax(outputs['logits'], dim=1)
            else:
                # Fallback during warmup phase
                if isinstance(outputs, dict) and 'projections' in outputs:
                    predictions = torch.zeros(outputs['projections'].shape[0], 
                                           dtype=torch.long, device=self.device)
                else:
                    # If outputs is neither tensor nor dict with projections, return zeros
                    predictions = torch.zeros(1, dtype=torch.long, device=self.device)
        
        return predictions
    
    def forward(self, x):
        training_phase = self._get_training_phase()
        mode = 'contrast' if training_phase == 'warmup' else 'joint'
        return self.model(x, mode=mode)
    
    def training_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        losses = self._compute_losses(outputs, labels)
        
        # Log losses
        training_phase = self._get_training_phase()
        self.log('train/total_loss', losses['total'], on_step=True, on_epoch=True, batch_size=len(images))
        self.log('train/phase', float(['warmup', 'joint', 'finetune'].index(training_phase)), on_step=False, on_epoch=True)
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':
                self.log(f'train/{loss_name}_loss', loss_value, on_step=True, on_epoch=True, batch_size=len(images))
        
        # Compute metrics if we have classification outputs
        if isinstance(outputs, dict) and 'logits' in outputs:
            with torch.no_grad():
                preds = self._predict(outputs)
                acc = (preds == labels).float().mean()
                mae = torch.abs(preds.float() - labels.float()).mean()
                
                self.log('train/accuracy', acc, on_step=True, on_epoch=True, batch_size=len(images))
                self.log('train/mae', mae, on_step=True, on_epoch=True, batch_size=len(images))
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        losses = self._compute_losses(outputs, labels)
        
        # Log losses
        self.log('val/total_loss', losses['total'], on_step=False, on_epoch=True, batch_size=len(images))
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':
                self.log(f'val/{loss_name}_loss', loss_value, on_step=False, on_epoch=True, batch_size=len(images))
        
        # Compute metrics if we have classification outputs
        if isinstance(outputs, dict) and 'logits' in outputs:
            with torch.no_grad():
                preds = self._predict(outputs)
                acc = (preds == labels).float().mean()
                mae = torch.abs(preds.float() - labels.float()).mean()
                
                self.log('val/accuracy', acc, on_step=False, on_epoch=True, batch_size=len(images))
                self.log('val/mae', mae, on_step=False, on_epoch=True, batch_size=len(images))
                
                # Balanced accuracy and per-class accuracy
                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                
                try:
                    bal_acc = balanced_accuracy_score(labels_np, preds_np)
                    self.log('val/balanced_accuracy', bal_acc, on_step=False, on_epoch=True, batch_size=len(images))
                    
                    # Per-class accuracy
                    for i in range(self.hparams.num_classes):
                        mask = labels_np == i
                        if mask.sum() > 0:
                            class_acc = (preds_np[mask] == i).mean()
                            self.log(f'val/class_{i}_accuracy', class_acc, on_step=False, on_epoch=True, batch_size=mask.sum())
                except Exception:
                    pass
        
        return losses['total']
    
    def test_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        losses = self._compute_losses(outputs, labels)
        
        # Log losses
        self.log('test/total_loss', losses['total'], on_step=False, on_epoch=True, batch_size=len(images))
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':
                self.log(f'test/{loss_name}_loss', loss_value, on_step=False, on_epoch=True, batch_size=len(images))
        
        # Compute metrics if we have classification outputs
        if isinstance(outputs, dict) and 'logits' in outputs:
            with torch.no_grad():
                preds = self._predict(outputs)
                acc = (preds == labels).float().mean()
                mae = torch.abs(preds.float() - labels.float()).mean()
                
                self.log('test/accuracy', acc, on_step=False, on_epoch=True, batch_size=len(images))
                self.log('test/mae', mae, on_step=False, on_epoch=True, batch_size=len(images))
                
                # Balanced accuracy and per-class accuracy
                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                
                try:
                    bal_acc = balanced_accuracy_score(labels_np, preds_np)
                    self.log('test/balanced_accuracy', bal_acc, on_step=False, on_epoch=True, batch_size=len(images))
                    
                    # Per-class accuracy  
                    for i in range(self.hparams.num_classes):
                        mask = labels_np == i
                        if mask.sum() > 0:
                            class_acc = (preds_np[mask] == i).mean()
                            self.log(f'test/class_{i}_accuracy', class_acc, on_step=False, on_epoch=True, batch_size=mask.sum())
                except Exception:
                    pass
        
        return losses['total']
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01  # Lower minimum for better convergence
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Update contrastive loss device if needed
        if self.contrastive_loss.device != self.device:
            self.contrastive_loss.device = self.device
            self.contrastive_loss.ordinal_distance_matrix = self.contrastive_loss.ordinal_distance_matrix.to(self.device)
            if self.contrastive_loss.prototypes is not None:
                self.contrastive_loss.prototypes.data = self.contrastive_loss.prototypes.data.to(self.device)
    
    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch."""
        # Ensure contrastive loss is on correct device
        if self.contrastive_loss.device != self.device:
            self.contrastive_loss.device = self.device
            self.contrastive_loss.ordinal_distance_matrix = self.contrastive_loss.ordinal_distance_matrix.to(self.device)
            if self.contrastive_loss.prototypes is not None:
                self.contrastive_loss.prototypes.data = self.contrastive_loss.prototypes.data.to(self.device)

class QWKLoss(nn.Module):
    """Quadratic Weighted Kappa loss for ordinal classification."""
    
    def __init__(self, num_classes: int, alpha: Optional[torch.Tensor] = None):
        super(QWKLoss, self).__init__()
        self.num_classes = num_classes
        self.cost_matrix = self._create_cost_matrix(num_classes)
        
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = alpha
    
    def _create_cost_matrix(self, num_classes: int) -> torch.Tensor:
        """Create quadratic cost matrix for QWK."""
        cost_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                cost_matrix[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        return torch.tensor(cost_matrix, dtype=torch.float32)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute QWK loss.
        
        Args:
            y_pred: Predicted probabilities [batch_size, num_classes]
            y_true: True labels [batch_size]
        """
        device = y_pred.device
        self.cost_matrix = self.cost_matrix.to(device)
        self.alpha = self.alpha.to(device)
        
        # Convert labels to one-hot
        if y_true.dim() == 1:
            y_true_onehot = F.one_hot(y_true, 
                                     num_classes=self.num_classes).float()
        else:
            y_true_onehot = y_true.float()
        
        # Get costs for each true class
        targets = torch.argmax(y_true_onehot, dim=1)
        costs = self.cost_matrix[targets]
        
        # Apply class weights
        weighted_pred = y_pred * self.alpha.unsqueeze(0)
        weighted_pred = weighted_pred / weighted_pred.sum(dim=1, keepdim=True)
        
        # Compute QWK loss
        numerator = torch.sum(costs * weighted_pred)
        
        # Normalization
        sum_prob = torch.sum(weighted_pred, dim=0)
        n = torch.sum(y_true_onehot, dim=0)
        
        cost_sum_prob = torch.matmul(
            self.cost_matrix, sum_prob.unsqueeze(1)
        ).squeeze()
        n_normalized = n / (torch.sum(n) + 1e-9)
        
        denominator = torch.sum(cost_sum_prob * n_normalized) + 1e-9
        
        loss = numerator / denominator
        
        return loss


import os

def visualize_embeddings(model, dataloaders: dict, device, save_dir, 
                        class_name, wandb_logger=None):
    import os
    model.eval()
    model = model.to(device)
    all_embeddings = []
    all_labels = []
    all_splits = []
    split_names = list(dataloaders.keys())
    for split in split_names:
        embeddings = []
        labels = []
        loader = dataloaders[split]
        with torch.no_grad():
            for batch in loader:
                imgs = batch["img"].to(device)
                lbls = batch["label"].cpu().numpy()
                features = model.model.backbone(imgs)
                x = model.model.projection_head[0](features)
                emb = x.cpu().numpy()
                embeddings.append(emb)
                labels.append(lbls)
        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            all_splits.extend([split]*len(labels))
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_splits = np.array(all_splits)

    # PCA
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(all_embeddings)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', 
                learning_rate='auto')
    emb_tsne = tsne.fit_transform(all_embeddings)

    # Plot all splits together (color by label)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = sns.color_palette("colorblind", 
                               n_colors=len(np.unique(all_labels)))
    for ax, emb, title in zip(axes, [emb_pca, emb_tsne], ["PCA", "t-SNE"]):
        sns.scatterplot(
            x=emb[:, 0], y=emb[:, 1], hue=all_labels, palette=palette, 
            ax=ax, legend='full', s=30, alpha=0.8
        )
        ax.set_title(f"Embedding Visualization ({title}) - All Splits - {class_name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Label", loc="best")
    plt.tight_layout()
    save_path_all = os.path.join(save_dir, f"embeddings_all_{class_name}.png")
    plt.savefig(save_path_all, dpi=200)
    if wandb_logger is not None and hasattr(wandb_logger, 'experiment'):
        wandb_logger.experiment.log({"embeddings_all": wandb.Image(save_path_all)})
    plt.close(fig)

    # Plot val+test only
    mask = np.isin(all_splits, ["val", "test"])
    emb_pca_vt = emb_pca[mask]
    emb_tsne_vt = emb_tsne[mask]
    labels_vt = all_labels[mask]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = sns.color_palette("colorblind", 
                               n_colors=len(np.unique(labels_vt)))
    for ax, emb, title in zip(axes, [emb_pca_vt, emb_tsne_vt], 
                              ["PCA", "t-SNE"]):
        sns.scatterplot(
            x=emb[:, 0], y=emb[:, 1], hue=labels_vt, palette=palette, 
            ax=ax, legend='full', s=30, alpha=0.8
        )
        ax.set_title(f"Embedding Visualization ({title}) - Val+Test - {class_name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Label", loc="best")
    plt.tight_layout()
    save_path_vt = os.path.join(save_dir, f"embeddings_val_test_{class_name}.png")
    plt.savefig(save_path_vt, dpi=200)
    if wandb_logger is not None and hasattr(wandb_logger, 'experiment'):
        wandb_logger.experiment.log({"embeddings_val_test": wandb.Image(save_path_vt)})
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, split_name, save_dir, 
                         class_name, wandb_logger=None):
    import os
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix ({split_name}) - {class_name}')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_{split_name}_{class_name}.png')
    plt.savefig(save_path, dpi=200)
    if wandb_logger is not None and hasattr(wandb_logger, 'experiment'):
        wandb_logger.experiment.log({f'confusion_matrix_{split_name}': wandb.Image(save_path)})
    plt.close(fig)


class OrdinalFocalEMDLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training OrdinalDenseNet with a combination of FocalLoss and EMDLoss.
    This is a much simpler variant for ordinal classification.
    """
    def __init__(
        self,
        num_classes: int = 3,
        spatial_dims: int = 3,
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        # Focal loss parameters
        focal_gamma: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        emd_weight: float = 0.5,
        focal_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = OrdinalDenseNet(
            num_classes=num_classes,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
        )

        # Losses
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction='mean'
        )
        self.emd_loss = EMDLoss(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _compute_losses(self, logits, labels):
        loss_focal = self.focal_loss(logits, labels)
        loss_emd = self.emd_loss(logits, labels)
        total_loss = self.hparams.focal_weight * loss_focal + self.hparams.emd_weight * loss_emd
        return total_loss, loss_focal, loss_emd

    def training_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        logits = self.model(images)
        total_loss, loss_focal, loss_emd = self._compute_losses(logits, labels)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, batch_size=len(images))
        self.log('train/focal_loss', loss_focal, on_step=True, on_epoch=True, batch_size=len(images))
        self.log('train/emd_loss', loss_emd, on_step=True, on_epoch=True, batch_size=len(images))
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            mae = torch.abs(preds.float() - labels.float()).mean()
            self.log('train/accuracy', acc, on_step=True, on_epoch=True, batch_size=len(images))
            self.log('train/mae', mae, on_step=True, on_epoch=True, batch_size=len(images))
        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        logits = self.model(images)
        total_loss, loss_focal, loss_emd = self._compute_losses(logits, labels)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, batch_size=len(images))
        self.log('val/focal_loss', loss_focal, on_step=False, on_epoch=True, batch_size=len(images))
        self.log('val/emd_loss', loss_emd, on_step=False, on_epoch=True, batch_size=len(images))
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            mae = torch.abs(preds.float() - labels.float()).mean()
            self.log('val/accuracy', acc, on_step=False, on_epoch=True, batch_size=len(images))
            self.log('val/mae', mae, on_step=False, on_epoch=True, batch_size=len(images))
            # Balanced accuracy and per-class accuracy
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            try:
                bal_acc = balanced_accuracy_score(labels_np, preds_np)
                self.log('val/balanced_accuracy', bal_acc, on_step=False, on_epoch=True, batch_size=len(images))
                num_classes = self.hparams.num_classes if hasattr(self.hparams, 'num_classes') else logits.shape[1]
                for i in range(num_classes):
                    mask = labels_np == i
                    if mask.sum() > 0:
                        class_acc = (preds_np[mask] == i).mean()
                        self.log(f'val/class_{i}_accuracy', class_acc, on_step=False, on_epoch=True, batch_size=mask.sum())
            except Exception:
                pass
        return total_loss

    def test_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch["label"]
        logits = self.model(images)
        total_loss, loss_focal, loss_emd = self._compute_losses(logits, labels)
        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True, batch_size=len(images))
        self.log('test/focal_loss', loss_focal, on_step=False, on_epoch=True, batch_size=len(images))
        self.log('test/emd_loss', loss_emd, on_step=False, on_epoch=True, batch_size=len(images))
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            mae = torch.abs(preds.float() - labels.float()).mean()
            self.log('test/accuracy', acc, on_step=False, on_epoch=True, batch_size=len(images))
            self.log('test/mae', mae, on_step=False, on_epoch=True, batch_size=len(images))
        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }