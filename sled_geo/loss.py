from torch import nn, Tensor
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self,):
        super(MSELoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


class KLDivLoss(nn.Module):
    def __init__(self,):
        super(KLDivLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = F.log_softmax(input, dim=1)
        criterion = nn.KLDivLoss(reduction='batchmean')
        loss = criterion(input, target)
        return loss

#TODO: I cannot remember how to implement this.  I think it's
# - take probability logits among dimensions instead of among samples
# - do i need label probability in the same vein?
# - then InfoNCE
class SoftTargetLoss(nn.Module):
    def __init__(self,):
        super(SoftTargetLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = F.log_softmax(input, dim=1)
        criterion = nn.CrossEntropyLoss()
