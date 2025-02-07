import onnxruntime as ort 
import onnx 
import numpy as np 
import torch.nn as nn 
import torch 
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image
from torchvision import transforms
import os 

def quantize_eigenplaces_model(model: nn.Module, model_name: str): 
    model.eval() 
    dummy_input = torch.randn(1, 3, 512, 512)
    os.makedirs(os.path.join(os.path.dirname(__file__), "methods/weights/EigenPlaces"), exist_ok=True)
    onnx_path = os.path.join(os.path.dirname(__file__), "methods/weights/EigenPlaces", "tmp.onnx")
    torch.onnx.export(model,               # model being run
                 dummy_input,          # model input (or a tuple for multiple inputs)
                 onnx_path,            # where to save the model
                 export_params=True,   # store the trained parameter weights inside the model file
                 opset_version=11,     # the ONNX version to export the model to
                 do_constant_folding=True) 

    quantized_model_path = os.path.join(os.path.dirname(__file__), "methods/weights/EigenPlaces", f"{model_name}.onnx")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )


model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )



class EigenPlacesINT8Model(nn.Module): 
    def __init__(self, model_name: str): 
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), "methods/weights/EigenPlaces", f"{model_name}.onnx")
        self.model = onnx.load(path)
        self.device = "cpu"
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def forward(self, x: torch.Tensor): 
        x = x.detach().cpu().numpy()
        input_name = self.session.get_inputs()[0].name
        outputs = [] 
        for img in x: 
            outputs.append(self.session.run(None, {input_name: img[None, ...]})[0])
        return torch.tensor(np.vstack(outputs))
    
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
    
quantize_eigenplaces_model(model, "EigenPlaces-D2048-INT8")

# Example usage
if __name__ == "__main__":
    model = EigenPlacesINT8Model("EigenPlaces-D2048-INT8")
    input = torch.randn(10, 3, 512, 512) 
    out = model(input) 
    model.to("cpu")
    sd = model.state_dict()
    print(sd)
