import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import math

class LoRAParametrization(nn.Module):
    def __init__(self, lora_rank):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_A = None
        self.lora_B = None

    def initialize_lora(self, weight_shape):
        if self.lora_A is None or self.lora_B is None:
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            self.lora_A = nn.Parameter(torch.zeros(self.lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, weight):
        self.initialize_lora(weight.shape)
        delta_W = torch.matmul(self.lora_B, self.lora_A)
        return weight + delta_W

def add_lora_parametrizations_to_module(module, lora_rank):
    # Add LoRA to the module's own parameters (non-recursive)
    for param_name, param in list(module.named_parameters(recurse=False)):
        if param.ndim == 2:
            # Freeze the original parameter
            param.requires_grad = False
            # Check if the parameter is already parametrized
            if not parametrize.is_parametrized(module, param_name):
                # Register the LoRA parametrization
                parametrize.register_parametrization(
                    module, param_name, LoRAParametrization(lora_rank)
                )

def add_lora_adapters(model, lora_rank, submodule_name):
    # Directly access the submodule without recursion
    submodule = model
    for attr in submodule_name.split('.'):
        submodule = getattr(submodule, attr)
    # Add LoRA to the submodule's parameters
    add_lora_parametrizations_to_module(submodule, lora_rank)
    # Manually add LoRA to known child modules if necessary
    # For example, if the submodule has a 'layers' attribute that is a ModuleList
    if hasattr(submodule, 'layers'):
        for layer in submodule.layers:
            add_lora_parametrizations_to_module(layer, lora_rank)
            # If layers have further submodules (e.g., 'self_attn', 'mlp'), you can add LoRA to them as well
            if hasattr(layer, 'self_attn'):
                add_lora_parametrizations_to_module(layer.self_attn, lora_rank)
            if hasattr(layer, 'mlp'):
                add_lora_parametrizations_to_module(layer.mlp, lora_rank)
    return model