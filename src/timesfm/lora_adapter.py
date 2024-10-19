import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import copy
import math

class LoRAParametrization(nn.Module):
    def __init__(self, lora_rank):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_A = None
        self.lora_B = None

    def initialize_lora(self, weight_shape, device=None, dtype=None):
        if self.lora_A is None or self.lora_B is None:
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            device = device or torch.device('cpu')
            dtype = dtype or torch.float32
            self.lora_A = nn.Parameter(torch.zeros(self.lora_rank, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank, device=device, dtype=dtype))
            nn.init.normal_(self.lora_A, mean=0., std=1.)
            nn.init.zeros_(self.lora_B)

    def forward(self, weight):
        self.initialize_lora(weight.shape, weight.device, weight.dtype)
        delta_W = torch.matmul(self.lora_B, self.lora_A)
        return weight + delta_W

def add_lora_adapters(model, lora_rank, submodule_name):
    """
    Adds LoRA adapters to all 2D parameters in the specified submodule.
    Freezes all parameters in the model to prevent them from being updated during training.
    """
    from torch.nn.utils import parametrize

    # freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # find the specified submodule
    try:
        submodule = model.get_submodule(submodule_name)
    except AttributeError:
        # If get_submodule is not available (older PyTorch versions), use an alternative method
        submodule = dict(model.named_modules()).get(submodule_name)
        if submodule is None:
            raise ValueError(f"Submodule '{submodule_name}' not found in the model.")

    # recursively search all parameters in the submodule
    modules_with_params = set()
    # Create a list of parameters to avoid modifying the dict during iteration
    params = list(submodule.named_parameters(recurse=True))
    for name, param in params:
        if param.ndim == 2:
            # Find the module that owns this parameter
            module_name = '.'.join(name.split('.')[:-1])
            owning_module = submodule
            if module_name:
                # Attempt to get the owning module
                try:
                    owning_module = submodule.get_submodule(module_name)
                except AttributeError:
                    # Fallback for older PyTorch versions
                    owning_module = dict(submodule.named_modules()).get(module_name)
                    if owning_module is None:
                        continue  # Skip if module not found
            # Avoid re-parametrizing the same parameter
            param_name = name.split('.')[-1]
            if parametrize.is_parametrized(owning_module, param_name):
                continue
            # Apply LoRA parametrization
            param.requires_grad = False  # Ensure the original parameter is frozen
            parametrize.register_parametrization(
                owning_module, param_name, LoRAParametrization(lora_rank)
            )
            modules_with_params.add(owning_module)

    return model