from engine import Tensor
from .main import Module
import numpy as np

class Linear(Module):
    def __init__(
            self, in_features: int, out_features: int,
            use_bias: bool = True, dtype = None
    ):
        self.use_bias = use_bias
        self.weight = Tensor(
            np.random.uniform(-1, 1, (out_features, in_features)),
            dtype=dtype, requires_grad=True
        )
        if self.use_bias:
            self.bias = Tensor(
                np.random.uniform(-1, 1, (out_features,)),
                dtype=dtype, requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.T()
        if self.use_bias:
            output += self.bias
        return output