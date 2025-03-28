from typing import *
from ..engine import Tensor
from ..nn import Module
import numpy as np

class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
        self.eps = eps
        self.weight = Tensor(
            np.ones(self.normalized_shape),
            dtype = np.float32, requires_grad=True
        )

        self.bias = Tensor(
            self._d.zeros(self.normalized_shape),
            dtype=self._d.float32, requires_grad=True, use_np=self.use_np
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized_dims = tuple(range(np.ndim(x) - len(self.normalized_shape), np.ndim(x)))
        mean = x.mean(axis=normalized_dims, keepdims=True)
        var = np.mean(((x - mean) ** 2), axis=normalized_dims, keepdims=True)
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        return self.weight * x_norm + self.bias