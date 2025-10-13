import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, Union, Dict


def stable_sigmoid(t: torch.Tensor) -> torch.Tensor:
    """
    Stable sigmoid function that avoids overflow issues for large values of t.
    This function is used to compute the sigmoid of a tensor t, handling both positive
    and negative values in a numerically stable way.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor
    """
    idx = t > 0
    out = torch.zeros_like(t)
    out[idx] = 1.0 / (1 + torch.exp(-t[idx]))
    exp_t = torch.exp(t[~idx])
    out[~idx] = exp_t / (1.0 + exp_t)
    return out


class CLM(nn.Module):
    """
    Implementation of the cumulative link models from :footcite:t:`vargas2020clm` as a
    torch layer. Different link functions can be used, including logit, probit
    and cloglog.

    Parameters
    ----------
    num_classes : int
        The number of classes.
    link_function : str
        The link function to use. Can be ``'logit'``, ``'probit'`` or ``'cloglog'``.
    min_distance : float, default=0.0
        The minimum distance between thresholds

    Attributes
    ----------
    num_classes : int
        The number of classes.
    link_function : str
        The link function to use. Can be ``'logit'``, ``'probit'`` or ``'cloglog'``.
    min_distance : float
        The minimum distance between thresholds
    dist_ : torch.distributions.Normal
        The normal (0,1) distribution used to compute the probit link function.
    thresholds_b_ : torch.nn.Parameter
        The torch parameter for the first threshold.
    thresholds_a_ : torch.nn.Parameter
        The torch parameter for the alphas of the thresholds.


    Example
    ---------
    >>> import torch
    >>> from dlordinal.output_layers import CLM
    >>> inp = torch.randn(10, 5)
    >>> fc = torch.nn.Linear(5, 1)
    >>> clm = CLM(5, "logit")
    >>> output = clm(fc(inp))
    >>> print(output)
    tensor([[0.7944, 0.1187, 0.0531, 0.0211, 0.0127],
            [0.4017, 0.2443, 0.1862, 0.0987, 0.0690],
            [0.4619, 0.2381, 0.1638, 0.0814, 0.0548],
            [0.4636, 0.2378, 0.1632, 0.0809, 0.0545],
            [0.4330, 0.2419, 0.1746, 0.0893, 0.0612],
            [0.5006, 0.2309, 0.1495, 0.0716, 0.0473],
            [0.6011, 0.2027, 0.1138, 0.0504, 0.0320],
            [0.5995, 0.2032, 0.1144, 0.0507, 0.0322],
            [0.4014, 0.2443, 0.1863, 0.0988, 0.0691],
            [0.6922, 0.1672, 0.0838, 0.0351, 0.0217]], grad_fn=<CopySlices>)

    """

    def __init__(
        self,
        num_classes: int,
        link_function: Literal["logit", "probit", "cloglog"],
        min_distance: int = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.dist_ = torch.distributions.Normal(0, 1)

        self.thresholds_b_ = torch.nn.Parameter(
            data=torch.Tensor([0]), requires_grad=True
        )
        self.thresholds_a_ = torch.nn.Parameter(
            data=torch.Tensor([1.0 for _ in range(self.num_classes - 2)]),
            requires_grad=True,
        )

    def _convert_thresholds(self, b, a, min_distance):
        a = a**2
        a = a + min_distance
        thresholds_param = torch.cat((b, a), dim=0)
        th = torch.sum(
            torch.tril(
                torch.ones(
                    (self.num_classes - 1, self.num_classes - 1), device=a.device
                ),
                diagonal=0,
            )
            * torch.reshape(
                torch.tile(thresholds_param, (self.num_classes - 1,)).to(a.device),
                shape=(self.num_classes - 1, self.num_classes - 1),
            ),
            dim=(1,),
        )
        return th

    def _compute_z3(self, projected: torch.Tensor, thresholds: torch.Tensor):
        m = projected.shape[0]
        a = torch.reshape(torch.tile(thresholds, (m,)), shape=(m, -1))
        b = torch.transpose(
            torch.reshape(
                torch.tile(projected, (self.num_classes - 1,)), shape=(-1, m)
            ),
            0,
            1,
        )

        z3 = a - b
        return z3

    def _apply_link_function(self, z3):
        if self.link_function == "probit":
            a3T = self.dist_.cdf(z3)
        elif self.link_function == "cloglog":
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:  # 'logit'
            a3T = stable_sigmoid(z3)

        return a3T

    def _clm(self, projected: torch.Tensor, thresholds: torch.Tensor):
        projected = torch.reshape(projected, shape=(-1,))

        m = projected.shape[0]
        z3 = self._compute_z3(projected, thresholds)
        a3T = self._apply_link_function(z3)

        ones = torch.ones((m, 1), device=projected.device)
        a3 = torch.cat((a3T, ones), dim=1)
        a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]

        return a3

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        output: Tensor
            The output tensor.
        """

        thresholds = self._convert_thresholds(
            self.thresholds_b_, self.thresholds_a_, self.min_distance
        )

        return self._clm(x, thresholds)


class WKLoss(nn.Module):
    """
    Implements Weighted Kappa Loss, introduced by :footcite:t:`deLaTorre2018kappa`.
    Weighted Kappa is widely used in ordinal classification problems. Its value lies in
    :math:`[0, 2]`, where :math:`2` means the random prediction.

    The loss is computed as follows:

    .. math::
        \\mathcal{L}(X, \\mathbf{y}) =
        \\frac{\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\sum\\limits_{k=1}^N q_{k,i} ~ p_{y_k,j}}
        {\\frac{1}{N}\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\left( \\sum\\limits_{k=1}^N q_{k,i} \\right)
        \\left( \\sum\\limits_{k=1}^N p_{y_k, j} \\right)}

    where :math:`q_{k,j}` denotes the normalised predicted probability, computed as:

    .. math::
        q_{k,j} = \\frac{\\text{P}(\\text{y} = j ~|~ \\mathbf{x}_k)}
        {\\sum\\limits_{i=1}^J \\text{P}(\\text{y} = i ~|~ \\mathbf{x}_k)},

    :math:`p_{y_k,j}` is the :math:`j`-th element of the one-hot encoded true label
    for sample :math:`k`, and :math:`\\omega` is the penalisation matrix, defined
    either linearly or quadratically. Its elements are:

    - Linear: :math:`\\omega_{i,j} = \\frac{|i - j|}{J - 1}`
    - Quadratic: :math:`\\omega_{i,j} = \\frac{(i - j)^2}{(J - 1)^2}`

    Parameters
    ----------
    num_classes : int
        The number of unique classes in your dataset.
    penalization_type : str, default='quadratic'
        The penalization method for calculating the Kappa statistics. Valid options are
        ``['linear', 'quadratic']``. Defaults to 'quadratic'.
    epsilon : float, default=1e-10
        Small value added to the denominator division by zero.
    weight : Optional[torch.Tensor], default=None
        Class weights to apply during loss computation. Should be a tensor of size
        `(num_classes,)`. If `None`, equal weight is given to all classes.
    use_logits : bool, default=False
        If `True`, the `input` is treated as logits. If `False`, `input` is treated
        as probabilities. The behavior of the `input` affects its expected format
        (logits vs. probabilities).

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import WKLoss
    >>> num_classes = 5
    >>> input = torch.randn(3, num_classes)  # Predicted logits for 3 samples
    >>> target = torch.randint(0, num_classes, (3,))  # Ground truth class indices
    >>> loss_fn = WKLoss(num_classes)
    >>> loss = loss_fn(input, target)
    >>> print(loss)
    """

    num_classes: int
    penalization_type: str
    weight: Optional[torch.Tensor]
    epsilon: float
    use_logits: bool

    def __init__(
        self,
        num_classes: int,
        penalization_type: str = "quadratic",
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-10,
        use_logits=False,
    ):
        super(WKLoss, self).__init__()
        self.num_classes = num_classes
        self.penalization_type = penalization_type
        self.epsilon = epsilon
        self.weight = weight
        self.use_logits = use_logits
        self.first_forward_ = True

    def _initialize(self, input, target):
        # Define error weights matrix
        repeat_op = (
            torch.arange(self.num_classes, device=input.device)
            .unsqueeze(1)
            .expand(self.num_classes, self.num_classes)
        )
        if self.penalization_type == "linear":
            self.weights_ = torch.abs(repeat_op - repeat_op.T) / (self.num_classes - 1)
        elif self.penalization_type == "quadratic":
            self.weights_ = torch.square((repeat_op - repeat_op.T)) / (
                (self.num_classes - 1) ** 2
            )

        # Apply class weight
        if self.weight is not None:
            # Repeat weight num_classes times in columns
            tiled_weight = self.weight.repeat((self.num_classes, 1)).to(input.device)
            self.weights_ *= tiled_weight

    def forward(self, input, target):
        """
        Forward pass for the Weighted Kappa loss.

        This method computes the Weighted Kappa loss between the predicted and true labels.
        The loss is based on the weighted disagreement between predictions and true labels,
        normalised by the expected disagreement under independence.

        Parameters
        ----------
        input : torch.Tensor
            The model predictions. Shape: ``(batch_size, num_classes)``.
            If ``use_logits=True``, these should be raw logits (unnormalised scores).
            If ``use_logits=False``, these should be probabilities (rows summing to 1).

        target : torch.Tensor
            Ground truth labels.
            Shape:
            - ``(batch_size,)`` if labels are class indices.
            - ``(batch_size, num_classes)`` if already one-hot encoded.
            The tensor will be converted to float internally.

        Returns
        -------
        loss : torch.Tensor
            A scalar tensor representing the weighted disagreement between predictions
            and true labels, normalised by the expected disagreement.
        """

        num_classes = self.num_classes

        # Convert to onehot if integer labels are provided
        if target.dim() == 1:
            y = torch.eye(num_classes).to(target.device)
            target = y[target]

        target = target.float()

        if self.first_forward_:
            if not self.use_logits and not torch.allclose(
                input.sum(dim=1), torch.tensor(1.0, device=input.device)
            ):
                raise ValueError(
                    "When passing use_logits=False, the input"
                    " should be probabilities, not logits."
                )
            elif self.use_logits and torch.allclose(
                input.sum(dim=1), torch.tensor(1.0, device=input.device)
            ):
                raise ValueError(
                    "When passing use_logits=True, the input"
                    " should be logits, not probabilities."
                )

            self._initialize(input, target)
            self.first_forward_ = False

        if self.use_logits:
            input = torch.nn.functional.softmax(input, dim=1)

        hist_rater_a = torch.sum(input, 0)
        hist_rater_b = torch.sum(target, 0)

        conf_mat = torch.matmul(input.T, target)

        bsize = input.size(0)
        nom = torch.sum(self.weights_ * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(self.weights_ * expected_probs / bsize)

        return nom / (denom + self.epsilon)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    Reduces the relative loss for well-classified examples and focuses on hard
    examples.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImbalancedOrdinalContrastiveLoss(nn.Module):
    """
    Enhanced Ordinal Contrastive Loss combining strategies from "A Tale of Two Classes" 
    with ordinal classification adaptations for the LISA 2025 QC challenge.
    
    Supports multiple strategies:
    - supervised_minority: SupCon for minority, NT-Xent for majority classes
    - supervised_prototypes: Prototype-based alignment with conditional attraction
    - combined: Both strategies together
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        strategy: str = "supervised_minority",
        prototype_similarity_threshold: float = 0.5,
        minority_threshold: float = 0.2,
        ntxent_batch_structure: str = "deterministic",  # "pairs", "random", "deterministic"
        ordinal_weight_scale: float = 2.0,
        device: str = "cuda"
    ):
        super(ImbalancedOrdinalContrastiveLoss, self).__init__()
        
        self.num_classes = num_classes
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.strategy = strategy
        self.prototype_similarity_threshold = prototype_similarity_threshold
        self.minority_threshold = minority_threshold
        self.ntxent_batch_structure = ntxent_batch_structure
        self.ordinal_weight_scale = ordinal_weight_scale
        self.device = device
        
        # Create ordinal distance matrix
        self.ordinal_distance_matrix = self._create_ordinal_distance_matrix()
        
        # Track class frequencies for adaptive weighting
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # Prototypes will be initialized lazily
        self.prototypes = None
        
    def _create_ordinal_distance_matrix(self) -> torch.Tensor:
        """Create ordinal distance matrix where distance increases with ordinal separation."""
        distance_matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                # Normalized absolute distance for ordinality
                distance_matrix[i, j] = abs(i - j) / (self.num_classes - 1)
        return distance_matrix.to(self.device)
    
    def _initialize_prototypes(self, feature_dim: int) -> nn.Parameter:
        """Initialize prototypes for ordinal classes."""
        if self.num_classes == 2:
            # Binary case: use bipolar prototypes as in paper
            prototypes = torch.zeros(2, feature_dim)
            prototypes[0, 0] = 1.0  # Majority class
            prototypes[1, 0] = -1.0  # Minority class
        else:
            # Ordinal case: arrange on semicircle to preserve order
            prototypes = torch.zeros(self.num_classes, feature_dim)
            angles = torch.linspace(0, np.pi, self.num_classes)
            
            prototypes[:, 0] = torch.cos(angles)
            if feature_dim > 1:
                prototypes[:, 1] = torch.sin(angles)
            
            # Add small noise to remaining dimensions
            if feature_dim > 2:
                prototypes[:, 2:] = torch.randn(self.num_classes, feature_dim - 2) * 0.01
        
        # Normalize to unit sphere
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return nn.Parameter(prototypes.to(self.device), requires_grad=True)
    
    def update_class_statistics(self, labels: torch.Tensor):
        """Update running statistics of class frequencies."""
        for i in range(self.num_classes):
            self.class_counts[i] += (labels == i).sum().float()
        self.total_samples += labels.size(0)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for handling imbalance."""
        if self.total_samples == 0:
            return torch.ones(self.num_classes).to(self.device)
        
        frequencies = self.class_counts / self.total_samples
        frequencies = torch.clamp(frequencies, min=1e-8)
        weights = 1.0 / frequencies
        weights = weights / weights.sum() * self.num_classes
        return weights
    
    def _identify_minority_classes(self) -> torch.Tensor:
        """Identify minority classes based on frequency threshold."""
        if self.total_samples == 0:
            return torch.arange(self.num_classes)
        
        frequencies = self.class_counts / self.total_samples
        minority_mask = frequencies < self.minority_threshold
        return torch.where(minority_mask)[0]
    
    def _compute_supervised_minority_loss(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Core Supervised Minority strategy: SupCon for minority, NT-Xent for majority.
        """
        minority_classes = self._identify_minority_classes()
        minority_mask = torch.isin(labels, minority_classes)
        majority_mask = ~minority_mask
        
        total_loss = 0.0
        
        # SupCon loss for minority classes
        if minority_mask.any():
            minority_features = features[minority_mask]
            minority_labels = labels[minority_mask]
            minority_loss = self._compute_ordinal_supcon_loss(minority_features, minority_labels)
            total_loss += minority_loss
        
        # NT-Xent loss for majority classes
        if majority_mask.any():
            majority_features = features[majority_mask]
            majority_loss = self._compute_ordinal_ntxent_loss(
                majority_features, labels[majority_mask]
            )
            total_loss += majority_loss
            
        return total_loss
    
    def _compute_ordinal_supcon_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervised contrastive loss with ordinal distance weighting."""
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Apply ordinal distance weighting to negatives
        ordinal_weights = torch.ones_like(logits)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and labels[i] != labels[j]:
                    distance = self.ordinal_distance_matrix[labels[i].item(), labels[j].item()]
                    ordinal_weights[i, j] = 1 + distance * self.ordinal_weight_scale
        
        # Remove self-contrast
        logits_mask = torch.ones_like(mask).scatter_(1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * ordinal_weights * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Apply class weighting
        class_weights = self.get_class_weights()
        sample_weights = class_weights[labels.view(-1)]
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Weight by class importance
        weighted_loss = -mean_log_prob_pos * sample_weights
        loss = weighted_loss.mean() * (self.temperature / self.base_temperature)
        
        return loss
    
    def _compute_ordinal_ntxent_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """NT-Xent loss with ordinal distance weighting."""
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # Compute similarity matrix
        similarity_matrix = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Same class mask (excluding self)
        labels_expanded = labels.unsqueeze(1)
        same_class_mask = (labels_expanded == labels_expanded.T).float()
        eye_mask = torch.eye(batch_size).to(device)
        same_class_mask = same_class_mask * (1 - eye_mask)
        
        # Apply ordinal weighting to negatives
        ordinal_weights = torch.ones_like(logits)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and same_class_mask[i, j] == 0:
                    distance = self.ordinal_distance_matrix[labels[i].item(), labels[j].item()]
                    ordinal_weights[i, j] = 1 + distance * self.ordinal_weight_scale
        
        ordinal_weights = ordinal_weights * (1 - eye_mask)
        
        total_loss = 0.0
        num_valid_anchors = 0
        
        for i in range(batch_size):
            same_class_indices = torch.where(same_class_mask[i] > 0)[0]
            
            if len(same_class_indices) > 0:
                # Select positive based on strategy
                if self.ntxent_batch_structure == "deterministic":
                    # Most similar same-class sample
                    same_class_similarities = similarity_matrix[i, same_class_indices]
                    best_pos_idx = same_class_indices[torch.argmax(same_class_similarities)]
                else:
                    # Random same-class sample
                    best_pos_idx = same_class_indices[torch.randint(len(same_class_indices), (1,)).item()]
                
                pos_logit = logits[i, best_pos_idx]
                exp_logits_weighted = torch.exp(logits[i]) * ordinal_weights[i]
                exp_sum = exp_logits_weighted.sum()
                
                loss_i = -pos_logit + torch.log(exp_sum + 1e-12)
                total_loss += loss_i
                num_valid_anchors += 1
        
        return total_loss / max(num_valid_anchors, 1) * (self.temperature / self.base_temperature)
    
    def _compute_prototype_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute prototype alignment loss with conditional attraction."""
        if self.prototypes is None:
            self.prototypes = self._initialize_prototypes(features.shape[1])
        
        batch_size = features.shape[0]
        loss = 0.0
        
        for i in range(batch_size):
            class_idx = labels[i].item()
            prototype = self.prototypes[class_idx]
            
            # Compute similarity with class prototype
            similarity = torch.dot(features[i], prototype)
            
            # Only apply loss if similarity is below threshold
            if similarity < self.prototype_similarity_threshold:
                pos_sim = torch.exp(similarity / self.temperature)
                
                # Negative similarities with other prototypes
                neg_sims = []
                for j in range(self.num_classes):
                    if j != class_idx:
                        neg_sim = torch.dot(features[i], self.prototypes[j])
                        neg_sims.append(torch.exp(neg_sim / self.temperature))
                
                if neg_sims:
                    neg_sum = torch.stack(neg_sims).sum()
                    loss += -torch.log(pos_sim / (pos_sim + neg_sum + 1e-12))
        
        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=features.device)
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with strategy selection.
        
        Args:
            features: Normalized feature embeddings [batch_size, feature_dim]
            labels: Integer labels [batch_size]
            update_stats: Whether to update class statistics
        
        Returns:
            Combined loss value
        """
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Update class statistics
        if update_stats:
            self.update_class_statistics(labels)
        
        if self.strategy == "supervised_minority":
            return self._compute_supervised_minority_loss(features, labels)
        elif self.strategy == "supervised_prototypes":
            contrastive_loss = self._compute_ordinal_supcon_loss(features, labels)
            prototype_loss = self._compute_prototype_loss(features, labels)
            return contrastive_loss + 0.1 * prototype_loss
        elif self.strategy == "combined":
            minority_loss = self._compute_supervised_minority_loss(features, labels)
            prototype_loss = self._compute_prototype_loss(features, labels)
            return minority_loss + 0.1 * prototype_loss
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")







class EMDLoss(nn.Module):
    """
    Computes the squared Earth Mover's Distance (EMD) loss, also known as the
    Ranked Probability Score (RPS), for ordinal classification tasks.

    This implementation follows the formulation presented by :footcite:t:`houys16semd`.
    The squared EMD loss is equivalent to the RPS described in
    :footcite:t:`epstein1969scoring`. It serves as a proper scoring rule for ordinal
    outcomes, encouraging probabilistic predictions that are both accurate and calibrated.

    Errors farther from the true class are penalised more heavily, reflecting the ordinal
    structure of the target variable.

    Parameters
    ----------
    num_classes : int
        The number of ordinal classes (denoted as J).

    Examples
    --------
    >>> import torch
    >>> from dlordinal.losses import EMDLoss
    >>> loss_fn = EMDLoss(num_classes=5)
    >>> y_pred = torch.randn(8, 5)  # Predicted logits
    >>> y_true = torch.tensor([0, 1, 2, 3, 4, 3, 1, 0])  # Class indices
    >>> loss = loss_fn(y_pred, y_true)
    """

    def __init__(self, num_classes: int):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        """
        Computes the squared Earth Mover's Distance (Ranked Probability Score) between
        predictions and targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            A tensor of shape (N, J) containing predicted logits, where N is the batch size
            and J is the number of classes.

        y_true : torch.Tensor
            A tensor of shape (N,) containing the true class indices as integers in the
            range [0, J - 1].

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean squared EMD loss over the batch.
        """

        # One-hot encode true labels
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes)

        # Convert logits to probabilities
        y_pred_proba = torch.nn.functional.softmax(y_pred, dim=1)

        # Compute the CDFs
        pred_cdf = torch.cumsum(y_pred_proba, dim=1)
        true_cdf = torch.cumsum(y_true_one_hot, dim=1)

        # Compute the squared EMD
        emd = torch.sum((pred_cdf - true_cdf) ** 2, dim=1)
        return emd.mean()


class OrdinalDenseNet(nn.Module):
    """
    DenseNet with ordinal learning for the LISA 2025 QC challenge.
    Combines the DenseNet264 backbone with a linear layer for ordinal learning.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        spatial_dims: int = 3,
        in_channels: int = 1,
    ):
        super(OrdinalDenseNet, self).__init__()
        
        from monai.networks.nets import DenseNet264
        
        # DenseNet backbone
        self.backbone = DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            norm="batch",
            dropout_prob=0.2,
        )
        
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass
        """
        # Get logits from backbone
        logits = self.backbone(x)
        
        return logits


class OrdinalContrastiveDenseNet(nn.Module):
    """
    Enhanced DenseNet with integrated ordinal contrastive learning for LISA 2025 QC challenge.
    
    Features:
    - Joint training with multiple ordinal-aware losses
    - Ordinal-specific architectural components  
    - Progressive training strategy support
    - Multiple prediction heads for robust inference
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        spatial_dims: int = 3,
        in_channels: int = 1,
        projection_dim: int = 64,
        hidden_dim: int = 512,
        backbone_features: int = 128,
        ordinal_embedding_dim: int = 32,
        use_ordinal_head: bool = True,
        dropout_prob: float = 0.3
    ):
        super(OrdinalContrastiveDenseNet, self).__init__()
        
        from monai.networks.nets import DenseNet264
        
        self.num_classes = num_classes
        self.use_ordinal_head = use_ordinal_head
        self.backbone_features = backbone_features
        
        # DenseNet backbone
        self.backbone = DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=backbone_features,
            norm="batch",
            dropout_prob=dropout_prob * 0.7,  # Slightly less dropout in backbone
        )
        
        # Projection head for standard contrastive learning
        if projection_dim != backbone_features:
            self.projection_head = nn.Sequential(
                nn.Linear(backbone_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob * 0.3),
                nn.Linear(hidden_dim, projection_dim),
            )
        else:
            self.projection_head = nn.Identity()
        
        # Standard classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(backbone_features, num_classes),
        )
        
        # Ordinal-specific components
        if use_ordinal_head:
            # Ordinal embedding head - learns ordinal relationships
            self.ordinal_embedding_head = nn.Sequential(
                nn.Linear(backbone_features, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob * 0.2),
                nn.Linear(hidden_dim // 2, ordinal_embedding_dim),
            )
            
            # Ordinal classifier - uses cumulative probabilities
            self.ordinal_classifier = nn.Sequential(
                nn.Linear(backbone_features, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob * 0.4),
                nn.Linear(hidden_dim // 2, num_classes - 1),  # For cumulative probabilities
            )
    
    def _cumulative_to_class_probs(self, cumulative_logits: torch.Tensor) -> torch.Tensor:
        """Convert cumulative logits to class probabilities for ordinal classification."""
        cumulative_probs = torch.sigmoid(cumulative_logits)
        
        batch_size = cumulative_probs.shape[0]
        device = cumulative_probs.device
        
        # Add boundaries: P(y≤-1)=0, P(y≤K)=1
        padded_cumulative = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            cumulative_probs,
            torch.ones(batch_size, 1, device=device)
        ], dim=1)
        
        # Class probabilities: P(y=k) = P(y≤k) - P(y≤k-1)
        class_probs = padded_cumulative[:, 1:] - padded_cumulative[:, :-1]
        
        return class_probs
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'joint',
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified forward pass supporting multiple training modes.
        
        Args:
            x: Input tensor
            mode: 'contrast', 'classify', 'ordinal', or 'joint'
            return_features: Whether to return intermediate features
            
        Returns:
            Tensor for single mode, Dict for joint mode or when return_features=True
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Handle single output modes for backward compatibility
        if mode == 'contrast' and not return_features:
            projections = self.projection_head(features)
            return F.normalize(projections, p=2, dim=1)
        elif mode == 'classify' and not return_features:
            return self.classification_head(features)
        
        # For joint mode or when requesting features, return dict
        outputs = {}
        
        if return_features:
            outputs['backbone_features'] = features
        
        # Contrastive outputs
        if mode in ['contrast', 'joint']:
            projections = self.projection_head(features)
            outputs['projections'] = F.normalize(projections, p=2, dim=1)
        
        # Classification outputs  
        if mode in ['classify', 'joint']:
            outputs['logits'] = self.classification_head(features)
        
        # Ordinal-specific outputs
        if mode in ['ordinal', 'joint'] and self.use_ordinal_head:
            # Ordinal embeddings for ordinal contrastive loss
            ordinal_embeddings = self.ordinal_embedding_head(features)
            outputs['ordinal_embeddings'] = F.normalize(ordinal_embeddings, p=2, dim=1)
            
            # Cumulative probabilities for ordinal classification
            cumulative_logits = self.ordinal_classifier(features)
            outputs['cumulative_logits'] = cumulative_logits
            outputs['ordinal_probs'] = self._cumulative_to_class_probs(cumulative_logits)
        
        return outputs