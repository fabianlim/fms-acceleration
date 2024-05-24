from ..unsloth_utils import inject_to_model
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
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