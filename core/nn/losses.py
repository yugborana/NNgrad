from core.engine import Tensor, DEFAULT_MIN
from nn.main import Module

class L1Loss(Module):
    def __init__(self, reduction: str = "sum") -> None:
        self.reduction = reduction
    
    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        l1 = abs((pred - actual))

        if self.reduction == "sum": return l1
        elif self.reduction == "mean": return l1 / actual.shape[0]

        else: raise ValueError("Invalid Reduction Type")

class MSELoss(Module):
    def __init__(self, reduction: str = "sum") -> None:
        self.reduction = reduction

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        mse = ((pred - actual) ** 2).sum()

        if self.reduction == "sum": return mse
        elif self.reduction == "mean": return mse / actual.shape[0]

        else: raise ValueError("Invalid Reduction Type")

class BCELoss(Module):
    def __init__(self, eps: float = DEFAULT_MIN) ->None:
        self.eps = eps

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        a: Tensor = pred * actual.clip(self.eps, 1 - self.eps).log()
        b: Tensor = (1 - pred) * (1 - actual).clip(self.eps, 1 - self.eps).log()

        return -(a + b).sum() / pred.shape[0]
    
    
