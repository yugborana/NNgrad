from typing import List, Union, Any

from engine import Tensor
from ..nn import Module

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = list(modules)

    def append(self, module: Module):
        self.modules.append(module)

    def parameters(self) -> List[Tensor]:
        x = []
        for module in self.modules:
            x += module.parameters()

        return x
    
    def forward(self, x: Union[Tensor, Any]) -> Union[Tensor, Any]:
        for i in range(len(self.modules)):
            x = self.modules[i](x)

        return x