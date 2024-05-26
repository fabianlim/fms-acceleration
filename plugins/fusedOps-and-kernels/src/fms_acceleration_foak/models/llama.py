from ..unsloth_utils import inject_to_model
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.cross_entropy_loss import fast_cross_entropy_loss, Fast_CrossEntropyLoss
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
import torch

# class RMSLayernorm(torch.nn.Module):

# def fast_rms_layernorm(layernorm, X, gemma = False):
#     W   = layernorm.weight
#     eps = layernorm.variance_epsilon
#     out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
#     return out
# pass

def rms_layernorm_forward(self, hidden_states):
    # gemma = False
    # return Fast_RMS_Layernorm.apply(
    #     hidden_states, 
    #     self.weight, self.variance_epsilon, gemma
    # )
    return fast_rms_layernorm(
        self, hidden_states
    )

from transformers.models.llama.modeling_llama import LlamaRMSNorm

PATCH = {
    'layernorm': (
        rms_layernorm_forward, LlamaRMSNorm
    )
}

from torch.nn import CrossEntropyLoss

class FastCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # return super().forward(input, target)
        # return fast_cross_entropy_loss(input, target)
        loss = Fast_CrossEntropyLoss.apply(
            input, target
        )
        n_items = torch.count_nonzero(target != -100)
        return loss.sum() / n_items

from ..fused_ops.unsloth_lora.gptq.fast_lora import apply_lora_qkv, apply_lora_o

def create_qkv_functions(attn: torch.nn.Module):

    Q = K = V = None

    def _state(X):
        nonlocal Q, K, V
        if Q is None and K is None and V is None:
            # this one assumes inside is q_proj, k_proj, ...
            Q, K, V = apply_lora_qkv(attn, X)

    def _lin_q(self, X):
        nonlocal Q
        _state(X)
        assert Q is not None, "qkv out of sync"
        out, Q = Q, None
        return out

    def _lin_k(self, X):
        nonlocal K
        _state(X)
        assert K is not None, "qkv out of sync"
        out, K = K, None
        return out

    def _lin_v(self, X):
        nonlocal V
        _state(X)
        assert V is not None, "qkv out of sync"
        out, V = V, None
        return out

    return _lin_q, _lin_k, _lin_v

from transformers.models.llama.modeling_llama import LlamaAttention

# simple utility function to guess if its lora layer
def _is_loralayer(
    module: torch.nn.Module, names = ['lora_A', 'lora_B', 'base_layer']
):
    return all([hasattr(module, x) for x in names])

# maybe need to seperate out the o rule
# consider have a function to combine ModelPatcherRule
def trigger_qkv(module: torch.nn.Module):
    return (
        isinstance(module, LlamaAttention) and
        _is_loralayer(module.q_proj) and
        _is_loralayer(module.k_proj) and
        _is_loralayer(module.v_proj) and
        _is_loralayer(module.o_proj)
    )

# def trigger_o(module: torch.nn.Module):
#     return (
#         isinstance(module, LlamaAttention) and
#         _is_loralayer(module.o_proj)
#     )

from ..fused_ops.unsloth_lora.gptq.fast_lora import get_lora_parameters, unpack_gptqstate, LoRA_W

def apply_lora_o2(self, X):
    Oqstate, OA, OB, OS = get_lora_parameters(self)
    O = LoRA_W.apply(X, *unpack_gptqstate(Oqstate), OA, OB, OS)
    return O

from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger

# FIXME: we need to be able to trigger on the 
# name of the module as well
def create_qkv_functions2(attn: torch.nn.Module):

    qkv = [
        (
            ModelPatcherTrigger(
                check=_is_loralayer, module_name=name
            ), _check
        )
        for name, _check in zip(
            ['q_proj', 'k_proj', 'v_proj'],
            create_qkv_functions(attn)
        )
    ] 
    if _is_loralayer(attn.o_proj):
        qkv.append(
            (
                ModelPatcherTrigger(
                    check=_is_loralayer, 
                    module_name='o_proj'
                ), apply_lora_o2
            )
        )
    return qkv

# def create_o_functions(attn: torch.nn.Module):
# 
#     return [
#         (
#             ModelPatcherTrigger(
#                 check=_is_loralayer, module_name='o_proj'
#             ), apply_lora_o
#         )
#     ]

ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-rms', # trigger=LlamaRMSNorm, 
        trigger=ModelPatcherTrigger(
            check=LlamaRMSNorm
        ),
        forward=fast_rms_layernorm
    ),
)
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-qkv', # trigger=trigger_qkv, 
        trigger=ModelPatcherTrigger(
            check=trigger_qkv,
        ),
        forward_builder=create_qkv_functions2
    )
)
# ModelPatcher.register(
#     ModelPatcherRule(
#         rule_id='llama-o', 
#         trigger=ModelPatcherTrigger(
#             check=trigger_o,
#         ),
#         forward_builder=create_o_functions
#     )
# )