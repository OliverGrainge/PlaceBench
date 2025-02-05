import onnxruntime as ort 
import onnx 
import numpy as np 
import torch.nn as nn 
import torch 
from onnx import numpy_helper

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
    
    weights_dict = {}

    def state_dict(self):
        weights_dict = {}
        for initializer in self.model.graph.initializer:
            name = initializer.name
            tensor = numpy_helper.to_array(initializer)
            weights_dict[name] = tensor
        return weights_dict

model_path = "/Users/olivergrainge/Documents/github/PlaceBench/methods/weights/PruneVPR/ResNet34_NetVLAD_agg_0.23_sparsity_0.297_R1_0.912.onnx" 



model = NetVLAD_SP_Model(model_path)
model.train()
print(model.state_dict())
input = torch.rand(20, 3, 320, 320)
output = model(input)
print(output.shape)
print(output[0].norm())