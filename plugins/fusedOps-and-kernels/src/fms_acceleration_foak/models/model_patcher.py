from dataclasses import dataclass, asdict
from typing import Type, Callable, Union, List, Dict, Tuple, Set
import torch
from types import MethodType
import importlib, inspect
import pandas as pd

# will be either a 
# - module class, which triggers on isinstance
# - callable, which will be useful to trigger on custom checks
# - (consider): adding a regex will will apply on the name
# ModelPatcherTrigger = Union[
#     torch.nn.Module, # trigger on isinstance
#     Callable[[torch.nn.Module], bool] # trigger on callable
# ]

from enum import Enum
class ModelPatcherTriggerType(Enum):
    module = 1
    callable = 2

@dataclass
class ModelPatcherTrigger:

    check: Union[
        torch.nn.Module, # trigger on isinstance
        Callable[[torch.nn.Module], bool] # trigger on callable
    ]

    type: ModelPatcherTriggerType = None

    module_name: str = None

    def is_triggered(
        self, module: torch.nn.Module,
        module_name: str,
    ):

        if (
            self.module_name is not None and
            module_name != self.module_name
        ):
            return False


        if (
            self.type == ModelPatcherTriggerType.module 
            and isinstance(module, self.check)
        ):
            return True

        try:
            # the function call may raise
            if (
                self.type == ModelPatcherTriggerType.callable
                and self.check(module)
            ):
                return True
        except:
            pass

        return False

    def __post_init__(self):

        if self.type is None:
            if (
                inspect.isclass(self.check) and 
                issubclass(self.check, torch.nn.Module)
            ):
                self.type = ModelPatcherTriggerType.module
            else:
                self.type = ModelPatcherTriggerType.callable

ModelForward = Callable

@dataclass
class ModelPatcherRule:
    # id, must be unique
    rule_id: str 

    # trigger
    trigger: ModelPatcherTrigger

    # trigger type
    # trigger_type: ModelPatcherTriggerType = None

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


# helpful to keep a history of all patching that has been done
@dataclass
class ModelPatcherHistory:
    # instance id of the class that was patched
    instance: int

    # class of the torch.nn.Module that was patched
    cls: str

    # parent class of the torch.nn.Module that was patched
    parent_cls: str

    # module name
    module_name: str

    # parent
    parent_module_name: str

    # name of the rule that was applied
    rule_id: str


# ------------------------ helpers -----------------------

# singleton class for patching models
class ModelPatcher:

    # singleton history of patches
    history: List[ModelPatcherHistory] = []

    # singleton list of rules that have been registered
    rules: Dict[str, ModelPatcherRule] = {}
    
    @staticmethod
    def load_patches(module_names: List[str], reload: bool = False):
        # each patch should be in a module that calls
        # ModelPatcher.register. So these will search
        # and load all the modules it can find

        # reload will trigger the register in that module
        for plugin_name in module_names:
            if importlib.util.find_spec(plugin_name):
                m = importlib.import_module(plugin_name)
                if reload:
                    try:
                        importlib.reload(m)
                    except AssertionError as e:
                        # this is if it was loaded already
                        pass

    @staticmethod
    def register(rule: ModelPatcherRule):
        # raise if added rule in duplicity
        assert rule.rule_id not in ModelPatcher.rules, \
            f"patch rule '{rule.rule_id}' already exists" 

        ModelPatcher.rules[rule.rule_id] = rule

    @staticmethod
    def is_triggered(module: torch.nn.Module, module_name: str):
        for name, rule in ModelPatcher.rules.items():

            if rule.trigger.is_triggered(module, module_name):
                return name, rule
            # trigger = rule.trigger
            # trigger_type = rule.trigger_type
            # if (
            #     trigger_type == ModelPatcherTriggerType.module 
            #     and isinstance(module, trigger)
            # ):
            #     return name, rule
            # try:
            #     # the function call may raise
            #     if (
            #         trigger_type == ModelPatcherTriggerType.callable
            #         and trigger(module)
            #     ):
            #         return name, rule
            # except:
            #     pass

        return None, None

    @staticmethod
    def patch(
        model: torch.nn.Module, 
        visited: Set = None,
        parent_prefix: str = None,
        parent_mcn: str  = None,
    ):
        # NOTE: should we avoid repatching?

        if visited is None:
            visited = set()

        for name, mod in model.named_modules():

            # some stats
            mod_id = id(mod)
            mod_class_name = mod.__class__.__name__
            name = name.split('.')
            if len(name) > 2:
                parent_module_name, module_name = '.'.join(name[:-1]), name[-1]
                parent_mod = model.get_submodule(parent_module_name)
                parent_mod_class_name = parent_mod.__class__.__name__
            else:
                # patching on model itself
                module_name = name[0]
                parent_mod_class_name = parent_module_name = ''
                if parent_prefix is not None:
                    parent_module_name = parent_prefix + '.' + parent_module_name
                if parent_mcn is not None:
                    parent_mod_class_name = parent_mcn

            rule_id, rule = ModelPatcher.is_triggered(mod, module_name)
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
                ModelPatcher.rules = {}
                for i, (trig, forw) in enumerate(forward):
                    ModelPatcher.register(ModelPatcherRule(
                        rule_id=f'{rule_id}-{i+1}',
                        trigger=trig, 
                        forward=forw,
                    ))

                # this is an isolated patch
                ModelPatcher.patch(
                    mod, visited=visited,
                    parent_prefix=parent_module_name,
                    parent_mcn=parent_mod_class_name,
                )

                # replace the rules
                ModelPatcher.rules = old_rules

                # done
                continue
            
            # otherwise
            mod.forward = MethodType(forward, mod)
            ModelPatcher.history.append(
                ModelPatcherHistory(
                    instance=mod_id, cls=mod_class_name, 
                    parent_cls=parent_mod_class_name,
                    module_name=module_name,
                    parent_module_name=parent_module_name,
                    rule_id=rule_id
                )
            )

    @staticmethod
    def summary():
        return pd.DataFrame([
            asdict(entry) for entry in ModelPatcher.history
        ])


