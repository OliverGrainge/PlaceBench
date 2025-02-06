import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from onnx import numpy_helper
import onnxruntime as ort 
import onnx 
from .base import SingleStageMethod


class NetVLAD_SP_Model(nn.Module): 
    def __init__(self, model_path): 
        super().__init__()
        self.model = onnx.load(model_path)
        # Configure session with specific provider
          # Will fall back to CPU if CUDA is not available
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def forward(self, inputs): 
        inputs = inputs.detach().cpu().numpy()
        result = []
        for input in inputs: 
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: input[None, ...]})[0]
            result.append(outputs)
        return torch.tensor(np.vstack(result))
    
    def to(self, device):
        available_providers = self.session.get_providers()
        
        if device == "cuda":
            if 'CUDAExecutionProvider' not in available_providers:
                raise RuntimeError("CUDA execution provider is not available. Please ensure CUDA is properly installed.")
            self.session.set_providers(['CUDAExecutionProvider'])
        else:
            self.session.set_providers(['CPUExecutionProvider'])
            
        self.device = device
        return self
    

    def state_dict(self):
        weights_dict = {}
        for initializer in self.model.graph.initializer:
            name = initializer.name
            tensor = numpy_helper.to_array(initializer)
            weights_dict[name] = tensor
        return weights_dict
    




class NetVLAD_SP(SingleStageMethod):
    def __init__(
        self,
        name="NetVLAD-SP",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (320, 320),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=7552,
        search_dist="cosine",
    ):
        model_path = os.path.join(os.path.dirname(__file__), "weights/PruneVPR", "ResNet34_NetVLAD_agg_0.23_sparsity_0.297_R1_0.912.onnx")
        model = NetVLAD_SP_Model(model_path)
        super().__init__(name, model, transform, descriptor_dim, search_dist)

