import numpy as np
from typing import *

MIN = 1e-6
MAX = 1 - 1e-6

Array = Union[np.ndarray]

class Tensor:

    grad_is_enabled: bool = True
    
    def __init__(self,
                 data: Union[Array, Any],
                 dtype = None,
                 children: tuple = (),
                 requires_grad: bool = False,
                 op = None
                ) -> None:
        self.dtype = dtype or np.float32
        self.op = op

        self.data = (
            np.array(data, self.dtype) if not isinstance(data, Array)
            else data.astype(dtype = self.dtype)
        )
        
        self.requires_grad = requires_grad
        self.prev = set([c for c in children if c.requires_grad])
        
        self.grad = (
            np.zeros_like(self.data) if self.requires_grad and self.grad_is_enabled
            else None
        )

        self.grad_fn = None
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    def reset_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def set_requires_grad(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise ValueError("Value not boolean")
        
        if self.grad is None and val == True:
            self.reset_grad
        
        self.requires_grad = val

    def backward(self) -> None:
        if not self.grad_is_enabled:
            raise ValueError("Gradient Calculation is diabled")
        
        order = []

        visited = set()
        recursion_stack = set()

        def toposort(node: "Tensor"):
            if node in recursion_stack:
                raise ValueError("Graph has cycle")
            if node not in visited:
                visited.add(node)
                recursion_stack.add(node)

                for child in node.prev:
                    toposort(child)

                recursion_stack.remove(node)
                order.append(node) 
        
        toposort(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(order):
            if node.grad_fn is not None:
                node.grad_fn()

    def clip(
            self, 
            min: float = MIN,
            max: float = MAX,
            clip_grad: bool = False,
            grad_min: float = MIN,
            grad_max: float = MAX
    ) -> "Tensor":
        
        self.data = self.clip(self.data, min, max)

        if clip_grad and self.grad is not None:
            self.grad = self.clip(self.grad, grad_min, grad_max)

    def __setitem__(self, indices, other):
        "Update specific elements of one tensor (self) using another tensor (other)"
        assert isinstance(other, Tensor)

        self.data[indices] = other.data.astype(self.data.dtype).copy()

        if self.grad is not None:
            if other.grad is not None:
                self.grad[indices] = other.grad.astype(self.grad.dtype).copy()
            else:
                self.grad[indices] = np.zeros_like(self.grad[indices])

    def __getitem__(self, indices):
        output = Tensor(
            self.data[indices], dtype=self.dtype,
            children= (self, ), op = "getitem",
            requires_grad= self.requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def out_backward():
                self.grad[indices] += output.grad
            
            output.backward = out_backward()
            output.requires_grad = True
        
        return output
    
    def broadcast(self, shape: Tuple[int]) -> "Tensor":
        data = self.broadcast(self.data, shape)
        
    



