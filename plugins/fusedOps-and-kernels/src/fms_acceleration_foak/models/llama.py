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
        print ('patched ent')
        # return super().forward(input, target)
        # return fast_cross_entropy_loss(input, target)
        loss = Fast_CrossEntropyLoss.apply(
            input, target
        )
        n_items = torch.count_nonzero(target != -100)
        return loss.sum() / n_items
