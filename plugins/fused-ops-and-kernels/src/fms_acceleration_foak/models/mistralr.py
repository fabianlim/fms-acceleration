# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from functools import partial

# Third Party
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRMSNorm,
    MistralMLP
)

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding as _fast_rope_embedding
from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger
from .model_patcher import combine_triggers, combine_functions
from .utils2 import build_lora_fused_ops, trigger_fused_ops
from .utils2 import KEY_QKV, KEY_O, KEY_MLP


# NOTE: fast_rope_embedding does not work with position_ids
# currently they are ignored
def fast_rope_embedding(Q, K, cos, sin, position_ids=None):
    return _fast_rope_embedding(Q, K, cos, sin)


# - do regex on RMSNorm class name
# - check on the tensors required for fast_rms_layernorm
ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-rms",
        trigger=ModelPatcherTrigger(check=MistralRMSNorm),
        forward=fast_rms_layernorm,
    ),
)

ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-qkvo",
        trigger=combine_triggers(
            ModelPatcherTrigger(
                check=partial(
                    trigger_fused_ops, attn_cls=MistralAttention,
                    submodule_names=["q_proj", "k_proj", "v_proj"],
                )
            ),
            ModelPatcherTrigger(
                check=partial(
                    trigger_fused_ops, attn_cls=MistralAttention,
                    submodule_names=["o_proj"],
                )
            ),
            logic='OR',
        ),
        forward_builder=combine_functions(
            partial(
                build_lora_fused_ops,
                submodule_names=["q_proj", "k_proj", "v_proj"],
                fused_op=KEY_QKV,
            ),
            partial(
                build_lora_fused_ops,
                submodule_names=["o_proj"],
                fused_op=KEY_O,
            ),
            logic='APPEND',
        ),
        forward_builder_args=["base_type"],
    )
)

ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-mlp",
        trigger=ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops, attn_cls=MistralMLP,
                submodule_names=["up_proj", "down_proj", "gate_proj"],
            )
        ),
        forward_builder= partial(
            build_lora_fused_ops,
            submodule_names=["up_proj", "down_proj", "gate_proj"],
            fused_op=KEY_MLP,
        ),
        forward_builder_args=["base_type"],
    )
)

ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-cross-ent",
        import_and_maybe_reload=(
            "torch.nn.CrossEntropyLoss",
            FastCrossEntropyLoss,
            "transformers.models.mistral.modeling_mistral",
        ),
    )
)

ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-rope",
        import_and_maybe_reload=(
            "transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb",
            fast_rope_embedding,
            None,
        ),
    )
)
