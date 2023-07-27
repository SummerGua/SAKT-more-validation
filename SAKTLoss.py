import torch.nn as nn
from torch import Tensor
import torch

class SAKTLoss(nn.Module):
    def __init__(self):
        super(SAKTLoss, self).__init__()

    def forward(self, pred: Tensor, truth: Tensor, mask: Tensor):
        mask=mask
        mask = mask.gt(0).view(-1) # view(-1): to one dimension

        pred = torch.masked_select(pred.view(-1), mask)
        truth = torch.masked_select(truth.view(-1), mask)

        loss = torch.nn.functional.binary_cross_entropy(
            pred, truth.float(), reduction="mean"
        )
        return loss