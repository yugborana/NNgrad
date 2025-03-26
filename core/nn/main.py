from typing import *
from engine import Tensor 

class Module:
    def __init__(self,) -> None:
        self.is_training = True

    def _get_tensors(self) -> List[Tensor]:
        tensors: List[Tensor] = []
        
        for _, value in self.__dict__.items():
            if isinstance(value, Tensor):
                tensors.append(value)

            elif isinstance(value, Module):
                tensors += value._get_tensors()
    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False

    def parameters(self) -> List[Tensor]:
        return (t for t in self._get_tensors if t.requires_grad)
    
    def zero_grad(self):
        for param in self.parameterS():
            param.reset_grad()

    def forward(self, *args:Any, **kwargs:Any):
        raise NotImplementedError("Base Class's Forward Method to be implemented")
    
    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        state_dict = {}
        for name, value in self.__dict__.items():
            pref = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor):
                state_dict[pref] = value
            elif isinstance(value, (Module, ModuleList, ModuleDict)):
                state_dict = state_dict | value.state_dict(prefix = pref)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
    

class ModuleList(Module):
    def __init__(self, modules: List[Module] = None):
        super().__init__()
        self.modules = modules or []

    def __getitem__(self, index: int) -> Module:
        return self.modules[index]
    
    def __setitem__(self, index: int, module: Module):
        self.modules[index] = module

    def __len__(self) -> int:
        return len(self.modules)
    
    def __iter__(self) -> Iterator[Module]:
        return iter(self.modules)
    
    def append(self, module: Module):
        return self.modules.append(module)
    
    def extend(self, modules: List[Module]):
        self.modules.extend(modules)\
        
    def insert(self, index: int, module: Module):
        return self.modules.insert(index, module)
    
    def parameters(self,) -> List[Tensor]:
        params = []
        for module in self.modules:
            params.extend(module.parameterS())
        return params
    
    def forward(self, x: Any) -> Any:
        for module in self.modules:
            x = module(x)
        return x
    

class ModuleDict(Module):
    def __init__(
            self, modules: Dict[str, Union[Module, ModuleList, 'ModuleDict']] = None,
    ):
        super().__init__()
        self.modules = modules

    def __init__(
            self, modules: Dict[str, Union[Module, ModuleList, 'ModuleDict']] = None,
            device: str = "gpu"
        ):
        super().__init__(device=device)
        self.modules = modules or {}

    def __getitem__(self, key: str) -> Union[Module, ModuleList, 'ModuleDict']:
        return self.modules[key]

    def __setitem__(self, key: str, module: Union[Module, ModuleList, 'ModuleDict']):
        self.modules[key] = module

    def __delitem__(self, key: str):
        del self.modules[key]

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self.modules)

    def __contains__(self, key: str) -> bool:
        return key in self.modules

    def clear(self):
        self.modules.clear()

    def pop(self, key: str) -> Union[Module, ModuleList, 'ModuleDict']:
        return self.modules.pop(key)

    def keys(self) -> List[str]:
        return list(self.modules.keys())

    def items(self) -> List[tuple]:
        return list(self.modules.items())

    def values(self) -> List[Union[Module, ModuleList, 'ModuleDict']]:
        return list(self.modules.values())

    def update(self, modules: Dict[str, Union[Module, ModuleList, 'ModuleDict']]):
        self.modules.update(modules)

    def parameters(self) -> List[Tensor]:
        params = []
        for module in self.modules.values():
            params.extend(module.parameters())
        return params
    
    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        sd = {}
        for k, mod in self.modules.items():
            pref = f"{prefix}.{k}" if prefix else k
            sd |= mod.state_dict(prefix=pref)
        return sd
    
    def forward(self, x: Any) -> Any:
        for module in self.modules.values():
            x = module(x)
        return x
    