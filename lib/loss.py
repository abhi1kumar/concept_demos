"""
    Custom loss functions
"""
import torch
import torch.nn as nn

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, y_pred, y_gt):
        assert y_pred.shape[1] == 2
        mean  = y_pred[:, 0]
        sigma_sq = y_pred[:, 1]               # batch,
        temp = torch.pow(mean- y_gt[:, 0], 2) # batch,
        loss = torch.mean(torch.log(sigma_sq) + temp/sigma_sq)

        return loss