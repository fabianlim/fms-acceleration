import torch
from typing import Type, Callable
from types import MethodType


# inspired by https://github.com/AutoGPTQ/AutoGPTQ/blob/
# ea829c7bbe83561c2b1de26795b6592992373ef7/auto_gptq/nn_modules/qlinear/
# __init__.py#L42
def inject_to_model(
    model: torch.nn.Module, target_module_type: Type, 
    builder: Callable,
):
    for name, m in model.named_modules():
        if not isinstance(m, target_module_type):
            continue
        new_m = builder(m)
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name[len(parent_name) + 1 :]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            parent = model
            child_name = name

        setattr(parent, child_name, new_m)

def patch_forward(
    model: torch.nn.Module, target_module_type: Type, 
    forward: Callable,
):
    for mod in model.modules():
        if isinstance(mod, target_module_type):
            mod.forward = MethodType(forward, mod)
