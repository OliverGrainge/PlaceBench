import torch

import config
from datasets import Essex3in1
from methods import TeTRA
from methods.tetra import Attention

method = TeTRA()
ds = Essex3in1(root=config.Essex3in1_root)
img_idx = 0

image = ds.query_paths[img_idx]
input_tensor = ds.transform(image)[None, ...]


"""
def get_attention_maps(tetra_method, input_tensor):
    attention_maps = []
    
    def attention_hook(module, input, output):
        if isinstance(module, Attention):
            # Get attention map from the attend module
            with torch.no_grad():
                q, k, v = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=module.heads),
                    module.to_qkv(module.lnorm1(input[0])).chunk(3, dim=-1)
                )
                attn_map = torch.matmul(q, k.transpose(-1, -2)) * module.scale
                attention_maps.append(attn_map.detach().cpu())
    
    # Register hooks on all attention layers
    hooks = []
    for module in tetra_method.model.modules():
        if isinstance(module, Attention):
            hooks.append(module.register_forward_hook(attention_hook))
    
    # Forward pass
    try:
        with torch.no_grad():
            output = tetra_method.model(input_tensor)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
    return output, attention_maps


attention = get_attention_maps(method, input_tensor)
print(attention)
"""
