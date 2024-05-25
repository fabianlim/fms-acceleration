from .model_patcher import ModelPatcher
from .llama import *

PATCHES = [
    '.models.llama'
]
PLUGIN_PREFIX = 'fms_acceleration_foak'
# TODO: remove the need for the prefix
ModelPatcher.load_patches(
    [f"{PLUGIN_PREFIX}{postfix}" for postfix in PATCHES]
)