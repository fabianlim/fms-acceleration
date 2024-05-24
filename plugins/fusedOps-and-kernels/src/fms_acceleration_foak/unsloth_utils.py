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

from typing import Any
import importlib
    
def patch_target_module(
    to_patch: str,
    replace_with: Any,
    target_module: str,
):
    to_patch = to_patch.split('.')
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    original_obj = getattr(source, obj_name_to_patch)
    setattr(source, obj_name_to_patch, replace_with)
    target_module = importlib.import_module(target_module)

    # reload and this should get the patched object
    importlib.reload(target_module)

    # replace it
    setattr(source, obj_name_to_patch, original_obj)
