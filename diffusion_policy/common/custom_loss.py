import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-7


class RMSELoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, actual):
        mse = F.mse_loss(pred, actual, reduction=self.reduction)
        rmse = torch.sqrt(mse + EPSILON)
        return rmse