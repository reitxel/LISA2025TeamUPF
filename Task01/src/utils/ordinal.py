import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ====================== CUMULATIVE LINK MODEL (CLM) LAYER ======================

class CumulativeLinkModel(nn.Module):
    """
    Fixed Cumulative Link Model matching Keras implementation more closely
    """
    
    def __init__(self, num_classes, link_function='logit', use_tau=True):
        super(CumulativeLinkModel, self).__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.use_tau = use_tau
        self.dist = Normal(0, 1)
        
        # Initialize thresholds matching Keras initialization
        b_init = torch.empty(1).uniform_(0, 0.1)
        a_min = np.sqrt((1.0 / (num_classes - 2)) / 2)
        a_max = np.sqrt(1.0 / (num_classes - 2))
        a_init = torch.empty(num_classes - 2).uniform_(a_min, a_max)
        
        self.thresholds_b = nn.Parameter(b_init)
        self.thresholds_a = nn.Parameter(a_init)
        
        if self.use_tau:
            # Initialize tau between 1 and 10 as in Keras
            self.tau = nn.Parameter(torch.empty(1).uniform_(1, 10))
        
        if self.link_function == 'glogit':
            self.lmbd = nn.Parameter(torch.ones(1))
            self.alpha = nn.Parameter(torch.ones(1))
            self.mu = nn.Parameter(torch.zeros(1))
    
    def _convert_thresholds(self, b, a):
        """Convert threshold parameters to ensure monotonicity."""
        a = torch.pow(a, 2)  # Ensure positive values
        thresholds_param = torch.cat([b, a], dim=0)
        ones_matrix = torch.tril(torch.ones(self.num_classes - 1, self.num_classes - 1, device=b.device))
        thresholds_param_tiled = thresholds_param.repeat(self.num_classes - 1)
        reshaped_param = thresholds_param_tiled.view(self.num_classes - 1, self.num_classes - 1)
        th = torch.sum(ones_matrix * reshaped_param, dim=1)
        return th
    
    def _nnpom(self, projected, thresholds):
        """Neural Network Proportional Odds Model computation."""
        if self.use_tau:
            # Clamp tau between 1 and 1000 as in Keras
            tau_clamped = torch.clamp(self.tau, min=1.0, max=1000.0)
            projected = projected.view(-1) / tau_clamped
        else:
            projected = projected.view(-1)
        
        batch_size = projected.shape[0]
        a = thresholds.repeat(batch_size, 1)
        b = projected.unsqueeze(1).repeat(1, self.num_classes - 1)
        z3 = a - b
        
        # Apply link function
        if self.link_function == 'probit':
            a3T = self.dist.cdf(z3)
        elif self.link_function == 'cloglog':
            a3T = 1 - torch.exp(-torch.exp(z3))
        elif self.link_function == 'glogit':
            a3T = 1.0 / torch.pow(1.0 + torch.exp(-self.lmbd * (z3 - self.mu)), self.alpha)
        else:  # 'logit'
            a3T = torch.sigmoid(z3)
        
        # Convert to probabilities
        ones = torch.ones(batch_size, 1, device=projected.device)
        a3 = torch.cat([a3T, ones], dim=1)
        a3 = torch.cat([a3[:, 0:1], a3[:, 1:] - a3[:, :-1]], dim=1)
        
        return a3
    
    def forward(self, x):
        thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
        return self._nnpom(x, thresholds)


# ====================== QUADRATIC WEIGHTED KAPPA (QWK) LOSS ======================

def make_cost_matrix(num_classes):
    """
    Create a quadratic cost matrix for ordinal classification.
    
    The cost increases quadratically with the distance between classes.
    For example, mistaking class 0 for class 3 has higher cost than
    mistaking class 0 for class 1.
    
    Parameters:
    -----------
    num_classes : int
        Number of ordinal classes
        
    Returns:
    --------
    cost_matrix : torch.Tensor
        Quadratic cost matrix of shape (num_classes, num_classes)
    """
    cost_matrix = np.reshape(np.tile(range(num_classes), num_classes), (num_classes, num_classes))
    cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_classes - 1) ** 2.0
    return torch.tensor(cost_matrix, dtype=torch.float32)


class QWKLoss(nn.Module):
    """
    Fixed Quadratic Weighted Kappa loss matching Keras implementation
    """
    
    def __init__(self, num_classes, alpha=None):
        super(QWKLoss, self).__init__()
        self.num_classes = num_classes
        self.cost_matrix = self.make_cost_matrix(num_classes)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def make_cost_matrix(self, num_classes):
        """Create quadratic cost matrix"""
        cost_matrix = np.reshape(np.tile(range(num_classes), num_classes), 
                                (num_classes, num_classes))
        cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_classes - 1) ** 2
        return torch.tensor(cost_matrix, dtype=torch.float32)
        
    def forward(self, y_pred, y_true):
        """
        Calculate QWK loss matching Keras implementation
        """
        device = y_pred.device
        self.cost_matrix = self.cost_matrix.to(device)
        self.alpha = self.alpha.to(device)
        
        # Convert integer labels to one-hot if needed
        if y_true.dim() == 1:
            y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        else:
            y_true_onehot = y_true.float()
        
        # Get the true class indices
        targets = torch.argmax(y_true_onehot, dim=1)
        
        # Get the costs for each true class
        costs = self.cost_matrix[targets]
        
        # Apply class weights to predictions
        weighted_pred = y_pred * self.alpha.unsqueeze(0)
        weighted_pred = weighted_pred / weighted_pred.sum(dim=1, keepdim=True)  # Renormalize
        
        # Compute weighted predictions - matching Keras
        numerator = torch.sum(costs * weighted_pred)
        
        # Compute normalization term - matching Keras implementation
        sum_prob = torch.sum(weighted_pred, dim=0)
        n = torch.sum(y_true_onehot, dim=0)
        
        # Reshape for matrix multiplication as in Keras
        cost_sum_prob = torch.matmul(self.cost_matrix, sum_prob.unsqueeze(1)).squeeze()
        n_normalized = n / torch.sum(n)
        
        epsilon = 1e-9
        denominator = torch.sum(cost_sum_prob * n_normalized) + epsilon
        
        # Compute final loss
        loss = numerator / denominator
        
        return loss


class QWKLossWithRegularization(nn.Module):
    """
    QWK Loss with optional regularization terms
    """
    def __init__(self, num_classes, ordinal_weight=0.1, entropy_weight=0.01):
        super().__init__()
        self.qwk_loss = QWKLoss(num_classes)
        self.ordinal_weight = ordinal_weight
        self.entropy_weight = entropy_weight
        
    def forward(self, y_pred, y_true):
        # Base QWK loss
        qwk = self.qwk_loss(y_pred, y_true)
        
        # Optional ordinal regularization
        if self.ordinal_weight > 0:
            pred_classes = torch.argmax(y_pred, dim=1)
            true_classes = y_true if y_true.dim() == 1 else torch.argmax(y_true, dim=1)
            ordinal_reg = torch.mean(torch.abs(pred_classes.float() - true_classes.float()))
            qwk = qwk + self.ordinal_weight * ordinal_reg
        
        # Optional entropy regularization
        if self.entropy_weight > 0:
            entropy = -torch.sum(y_pred * torch.log(y_pred + 1e-9), dim=1).mean()
            qwk = qwk + self.entropy_weight * entropy
            
        return qwk


class OrdinalFocalLoss(nn.Module):
    """Focal loss adapted for ordinal classification"""
    def __init__(self, num_classes, gamma=2.0, alpha=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        # Alpha weights for each class (can be inverse of class frequency)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, num_classes) - probabilities from CLM
        y_true: (batch_size,) - integer labels
        """
        device = y_pred.device
        self.alpha = self.alpha.to(device)
        
        # Get true class probabilities
        batch_size = y_pred.shape[0]
        class_mask = F.one_hot(y_true, self.num_classes).float()
        
        # Compute focal weights
        pt = torch.sum(y_pred * class_mask, dim=1)  # True class probability
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = self.alpha[y_true]
        
        # Compute focal loss
        ce_loss = -torch.log(pt + 1e-8)
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


class QWKFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, alpha=None, qwk_weight=0.7, focal_weight=0.3):
        super().__init__()
        self.qwk_loss = QWKLoss(num_classes)
        self.focal_loss = OrdinalFocalLoss(num_classes, gamma, alpha)
        self.qwk_weight = qwk_weight
        self.focal_weight = focal_weight
    
    def forward(self, y_pred, y_true):
        qwk = self.qwk_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        return self.qwk_weight * qwk + self.focal_weight * focal

# ====================== INTEGRATION WITH YOUR CODE ======================

class OrdinalDenseNet(nn.Module):
    """
    Fixed OrdinalDenseNet with proper BatchNorm
    """
    
    def __init__(self, num_classes=3, spatial_dims=3, in_channels=1):
        super(OrdinalDenseNet, self).__init__()
        
        from monai.networks.nets import DenseNet264
        
        # Create base DenseNet
        self.backbone = DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=1000,
            norm="batch",
            dropout_prob=0.2,
        )
        
        # Use BatchNorm1d instead of LayerNorm to match Keras
        self.fc = nn.Linear(1000, 1)
        self.bn = nn.BatchNorm1d(1)  # Changed from LayerNorm
        self.dropout = nn.Dropout(0.2)
        self.clm = CumulativeLinkModel(
            num_classes=num_classes,
            link_function='logit',
            use_tau=True
        )
        
    def forward(self, x):
        # Get features from backbone
        x = self.backbone(x)
        
        # Project to single dimension
        x = self.fc(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        # Apply CLM for ordinal probabilities
        x = self.clm(x)
        
        return x

# ====================== ADDITIONAL UTILITIES ======================

def ordinal_accuracy_off_by_k(y_true, y_pred, k=1):
    """
    Compute accuracy allowing predictions to be off by at most k classes.
    Useful metric for ordinal classification.
    
    Parameters:
    -----------
    y_true : torch.Tensor
        True labels as integers (batch_size,)
    y_pred : torch.Tensor
        Predicted probabilities (batch_size, num_classes)
    k : int
        Maximum allowed difference between prediction and true class
        
    Returns:
    --------
    accuracy : float
        Off-by-k accuracy
    """
    y_pred_class = torch.argmax(y_pred, dim=-1)
    
    diff = torch.abs(y_true - y_pred_class)
    correct = diff <= k
    
    return torch.mean(correct.float()).item()


def compute_ordinal_metrics(y_true, y_pred):
    """
    Compute various metrics for ordinal classification.
    
    Parameters:
    -----------
    y_true : torch.Tensor
        True labels as integers
    y_pred : torch.Tensor
        Predicted probabilities
        
    Returns:
    --------
    metrics : dict
        Dictionary containing various ordinal metrics
    """
    metrics = {}
    
    # Get predicted classes
    y_pred_class = torch.argmax(y_pred, dim=-1)
    
    # Log prediction distribution
    pred_counts = torch.bincount(y_pred_class, minlength=3)
    true_counts = torch.bincount(y_true, minlength=3)
    
    print("\nPrediction distribution:")
    for i in range(3):
        print(f"Class {i}: Predicted {pred_counts[i]}, True {true_counts[i]}")
    
    # Standard accuracy
    metrics['accuracy'] = (y_pred_class == y_true).float().mean().item()
    
    # Off-by-1 accuracy
    metrics['accuracy_off_by_1'] = ordinal_accuracy_off_by_k(y_true, y_pred, k=1)
    
    # Mean Absolute Error
    metrics['mae'] = torch.abs(y_true - y_pred_class).float().mean().item()
    
    # Log some example predictions
    print("\nExample predictions (first 5):")
    for i in range(min(5, len(y_true))):
        print(f"True: {y_true[i]}, Pred: {y_pred_class[i]}, "
              f"Probs: {y_pred[i].detach().cpu().numpy()}")
    
    # Quadratic Weighted Kappa
    from sklearn.metrics import cohen_kappa_score
    metrics['qwk'] = cohen_kappa_score(
        y_true.cpu().numpy(), 
        y_pred_class.cpu().numpy(), 
        weights='quadratic'
    )
    
    return metrics