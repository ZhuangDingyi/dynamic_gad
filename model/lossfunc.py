import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for classification tasks.
        
        :param alpha: Weighting factor for the class balance.
        :param gamma: Focusing parameter to reduce the relative loss for well-classified examples.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.
        
        :param inputs: Predictions (logits) from the model. Shape: [batch_size, num_classes].
        :param targets: Ground truth labels. Shape: [batch_size].
        :return: Computed Focal Loss.
        """
        # Convert targets to one-hot encoding
        targets = torch.eye(inputs.size(1),device=inputs.device)[targets]

        # Compute softmax over the inputs
        inputs_softmax = F.softmax(inputs, dim=1)

        # Compute the focal loss components
        pt = torch.sum(targets * inputs_softmax, dim=1)  # Probability of the target class
        log_pt = torch.log(pt+1e-5)
        focal_weight = (1 - pt) ** self.gamma

        # Compute the loss
        loss = -self.alpha * focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss