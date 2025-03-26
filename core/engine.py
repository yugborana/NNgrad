import numpy as np
from typing import *
from utils import broadcast_axis

DEFAULT_MIN = 1e-6
DEFAULT_MAX = 1 - 1e-6

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

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

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
        data = np.broadcast_to(self.data, shape)
        output = Tensor(
            data, dtype=self.dtype, children=(self, ), op="broadcast"
        )
        broadcasted_axis = broadcast_axis(self.shape, shape)[0]

        if self.requires_grad and self.grad_is_enabled:
            def broadcast_backward():
                self.grad += np.sum(output.grad, axis=broadcasted_axis)
            
            output.grad_fn = broadcast_backward
            output.set_requires_grad(True)

        return output
    
    def sum(self, axis: Union[int, Tuple[int]] = None, keepdims: bool = False) -> "Tensor":
        output = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            children=(self, ), op="sum"
        )

        if self.requires_grad and self.grad_is_enabled:
            #If keepdims=True, then the output keeps the same number of dimensions, and t_axis is set to None
            #If axis is specified and keepdims=False, t_axis stores the axis to be used later
            
            t_axis = axis if axis and not keepdims else None
            
            def sum_backward():
                if t_axis:
                    self.grad += np.ones_like(self.grad) * np.expand_dims(
                        output.grad, axis=t_axis
                    )
                else:
                    self.grad += np.ones_like(self.grad) * output.grad
            
            output.grad_fn = sum_backward
            output.set_requires_grad(True)

        return output
    

    def mean(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        n = self.data.size
        if isinstance(axis, int):
            n = self.data.shape[axis]
        if isinstance(axis, (tuple, list)):
            n = 1
            for dim in axis:
                n *= self.data.shape[dim]
        
        output: Tensor = self.sum(axis=axis, keepdims=keepdims) / n
        return output
    
    def T(self, axes: Iterable = None) -> "Tensor":
        output = Tensor(
            np.transpose(self.data, axes=axes),
            children=(self, ), op='T'
        )

        if self.requires_grad and self.grad_is_enabled:
            def t_backward():
                self.grad += np.transpose(output.grad, axes=axes)

            output.grad_fn = t_backward
            output.set_requires_grad(True)

        return output
    
    def exp(self) -> "Tensor":
        output = Tensor(
            np.exp(self.data), children=(self, ), op='exp'
        )

        if self.requires_grad and self.grad_is_enabled:
            def exp_backward():
                self.grad += output.data * output.grad

            output.grad_fn = exp_backward
            output.set_requires_grad(True)

        return output
    
    def log(self) -> 'Tensor':
        output = Tensor(
            np.log(self.data), children=(self, ), op='log'
        )

        if self.requires_grad and self.grad_is_enabled:
            def log_backward():
                self.grad += (output.grad / self.data)
            
            output.grad_fn = log_backward
            output.set_requires_grad(True)

        return output
    
    def reshape(self, shape: Tuple[int]) -> 'Tensor':
        output = Tensor(
            self.data.reshape(shape), children=(self, ), op='reshape'
        )

        if self.requires_grad and self.grad_is_enabled:
            def reshape_backward():
                self.grad += np.reshape(output.grad, shape=self.data.shape)
        
            output.grad_fn = reshape_backward
            output.set_requires_grad(True)
        
        return output
    
    def masked_fill(self, mask: Union[Array, list], value: Any) -> "Tensor":
        assert isinstance(mask, Array) or isinstance(mask, list)
        if isinstance(mask, list):
            mask = np.array(mask, dtype=np.int8)

        data = np.where(mask, np.array(value), self.data)
        output = Tensor(
            data, dtype=self.dtype, children=(self, ), op='mfill'
        )

        if self.requires_grad and self.grad_is_enabled:
            def mfill_backward():
                #Gradient of masked position made 0
                self.grad += np.where(mask, np.array(0), output.grad)

            output.grad_fn = mfill_backward
            output.set_requires_grad(True)

        return output
    
    def concat(self, others: List["Tensor"], dim: Optional[int] = 0) -> "Tensor":
        concat_list: List[Tensor] = [self]

        for other in others:
            assert isinstance(other, Tensor), f"Cannot concatenate type '{type(other)}'"
            concat_list.append(other)

        output = Tensor(
            np.concatenate([c.data for c in concat_list], axis=dim),
            children=tuple(concat_list), op='concat'
        )

        if self.grad_is_enabled:
            sizes = [c.shape[dim] for c in concat_list]
            sizes = np.array(sizes[:-1])
            splits = np.cumsum(sizes).tolist()
            if output.grad is None:
                    output.grad = np.zeros_like(output.data)
            grads = np.split(output.grad, splits, axis=dim)

            def concat_backward():
                for i, csor in enumerate(concat_list):
                    if csor.requires_grad:
                        csor.grad += grads[i]
            output.grad_fn = concat_backward
            output.set_requires_grad(True)
        
        return output
    
    def split(self, sections: int, dim: int = 0) -> List["Tensor"]:
        data: List[Array] = np.split(self.data, sections, axis=dim)
        

    


        
    



