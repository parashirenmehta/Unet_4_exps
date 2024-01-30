import torch
from torch import Tensor
from helper_functions.utils import dice_loss


class DiceLoss(torch.nn.Module):
    def __init__(self, multiclass: bool = True):
        super().__init__()
        self.multiclass = multiclass

    def forward(self, input: Tensor, target: Tensor):
        return dice_loss(input, target, self.multiclass)
