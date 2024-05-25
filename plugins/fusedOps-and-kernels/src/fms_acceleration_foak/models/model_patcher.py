from dataclasses import dataclass
from typing import Type, Callable, Union, List, Dict, Tuple, Set
import torch
from types import MethodType
import importlib

# will be either a 
# - module class, which triggers on isinstance
# - callable, which will be useful to trigger on custom checks
# - (consider): adding a regex will will apply on the name
ModelPatcherTrigger = Union[
    torch.nn.Module, # trigger on isinstance
    Callable[[torch.nn.Module], bool] # trigger on callable
]
ModelForward = Callable

from enum import Enum
class ModelPatcherTriggerType(Enum):
    module = 1
    callable = 2

@dataclass
class ModelPatcherRule:
    # id, must be unique
    rule_id: str 

    # trigger
    trigger: ModelPatcherTrigger

    # trigger type
    trigger_type: ModelPatcherTriggerType = None

    # takes in the torch module to build the forward.
    # will be helpful to
    # - do any pre-modification on the torch module

    # this is mutually exclusive from forward_builder
    forward: ModelForward = None
    
    # returns either
    # - a callable, which will be patched on the triggered module
    # - a list of trigger-forward tuples
    forward_builder: Callable[
        [torch.nn.Module], 
        Union[
            ModelForward, 
            List[Tuple[ModelPatcherTrigger, ModelForward]]
        ]
    ] = None

    def __post_init__(self):
        if (
            self.forward is not None and 
            self.forward_builder is not None
        ):
            raise ValueError(
                f"Rule '{self.rule_id}' cannot have both forward and "
                "foward builder specified."
            )

        if self.trigger_type is None:
            if issubclass(self.trigger, torch.nn.Module):
                self.trigger_type = ModelPatcherTriggerType.module
            else:
                self.trigger_type = ModelPatcherTriggerType.callable

# helpful to keep a history of all patching that has been done
@dataclass
class ModelPatcherHistory:
    # id of the class that was patched
    id: int

    # class of the torch.nn.Module that was patched
    cls: Type 

    # name of the rule that was applied
    rule_id: str

# ------------------------ helpers -----------------------

# def patch_forward(
#     model: torch.nn.Module, target_module_type: Type, 
#     forward: Callable,
# ):
#     for mod in model.modules():
#         if isinstance(mod, target_module_type):
#             mod.forward = MethodType(forward, mod)

def convert_forward_to_builder(forward: ModelForward):
    def _builder(_: torch.nn.Module):
        return forward
    return _builder

# singleton class for patching models
class ModelPatcher:

    # singleton history of patches
    history: List[ModelPatcherHistory] = []

    # singleton list of rules that have been registered
    rules: Dict[str, ModelPatcherRule] = {}

    # singleton boolean flag to reload patch modules
    reload_patch_modules: bool = False
    
    @staticmethod
    def load_patches(module_names: List[str]):
        # each patch should be in a module that calls
        # ModelPatcher.register. So these will search
        # and load all the modules it can find

        for plugin_name in module_names:
            if importlib.util.find_spec(plugin_name):
                m = importlib.import_module(plugin_name)
                if ModelPatcher.reload_patch_modules:
                    importlib.reload(m)

    @staticmethod
    def register(rule: ModelPatcherRule):
        # raise if added rule in duplicity
        assert rule.rule_id not in ModelPatcher.rules, \
            f"patch rule '{rule.rule_id}' already exists" 

        ModelPatcher.rules[rule.rule_id] = rule

    @staticmethod
    def is_trigger(module: torch.nn.Module):
        for name, rule in ModelPatcher.rules.items():

            trigger = rule.trigger
            trigger_type = rule.trigger_type
            if (
                (
                    trigger_type == ModelPatcherTriggerType.module 
                    and isinstance(module, trigger)
                ) 
                or
                (
                    trigger_type == ModelPatcherTriggerType.callable
                    and trigger(module)
                )
            ):
                return name, rule

        return None, None

    @staticmethod
    def patch(
        model: torch.nn.Module, 
        visited: Set = None
    ):
        if visited is None:
            visited = set()

        for mod in model.modules():

            # some stats
            mod_id = id(mod)
            mod_class_name = mod.__class__.__name__

            rule_id, rule = ModelPatcher.is_trigger(mod)
            if rule_id is None:
                continue

            # otherwise triggered
            if rule.forward is not None:
                forward = rule.forward
            else:
                forward = rule.forward_builder(mod)
            if isinstance(forward, list):
                # this will be list of tuples case

                # will descend down but
                # - clear old rules
                # - replace new rules
                old_rules = ModelPatcher.rules
                ModelPatcher.rules = []
                for i, (trig, forw) in enumerate(forward):
                    ModelPatcher.register(ModelPatcherRule(
                        rule_id=f'{rule_id}-{i+1}',
                        trigger=trig, 
                        forward_builder=convert_forward_to_builder(forw),
                    ))

                # this is an isolated patch
                ModelPatcher.patch(mod, visited=visited)

                # replace the rules
                ModelPatcher.rules = old_rules

                continue
            
            # otherwise
            mod.forward = MethodType(forward, mod)
            ModelPatcher.history.append(
                ModelPatcherHistory(id=mod_id, cls=mod_class_name, rule_id=rule_id)
            )


