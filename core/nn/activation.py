from typing import *
from core.engine import Tensor
import numpy as np

def relu(x: Tensor) -> Tensor:
    output = Tensor(
        np.maximum(0, x), dtype=x.dtype, children=(x, ),
        op='relu'
    )

    if x.requires_grad and Tensor.grad_is_enabled:
        def relu_backward():
            x.grad += (x.data > 0) * output.grad

        output.grad_fn = relu_backward
        output.set_requires_grad(True)

    return output

def tanh(x: Tensor) -> Tensor:
    output = Tensor(
        np.tanh(x.data), dtype=x.dtype, children=(x, ),
        op='tanh'
    )

    if x.requires_grad and Tensor.grad_is_enabled:
        def tanh_backward():
            x.grad += (1 - output.data**2) * output.grad

        output.grad_fn = tanh_backward
        output.set_requires_grad(True)

    return output

def sigmoid(x: Tensor) -> Tensor:
    e_x = x.exp(-x.data)
    output = Tensor(
        1 / (1 + e_x), dtype=x.dtype, children=(x, ),
        op='sigmoid'
    )

    if x.requires_grad and Tensor.grad_is_enabled:
        def sigmoid_backward():
            x.grad += (e_x / (1 + e_x) ** 2) * output.grad

        output.grad_fn = sigmoid_backward
        output.set_requires_grad(True)

    return output

def softmax(t: Tensor, axis: int = -1) -> Tensor:
    max_t = Tensor(
        np.max(t.data, axis=axis, keepdims=True),
        dtype=t.dtype
    )
    shifted_t = np.exp((t - max_t))
    shifted_exp_sum = np.sum(shifted_t)
    output = shifted_t / shifted_exp_sum
    
    return output