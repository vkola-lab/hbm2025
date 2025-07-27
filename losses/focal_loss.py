import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none').clamp(min=self.epsilon, max=1e9)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        loss = (alpha_t * (1 - pt).clamp(min=self.epsilon) ** self.gamma * ce_loss).mean()
        return loss