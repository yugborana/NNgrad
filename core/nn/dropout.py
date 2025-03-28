from typing import * 
from main import Module
from ..engine import Tensor
import numpy as np

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        if not self.is_training:
            return x
        
        mask = np.random.binomial(1, 1 - self.p, size=x.shape)

        mask = Tensor(
            mask.astype(np.int8), dtype=np.int8,
            requires_grad=False
        )

        output = (x * mask) / (1 - self.p)
        return output