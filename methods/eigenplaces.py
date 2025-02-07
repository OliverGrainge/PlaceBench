import os
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torchvision.transforms as T
from onnx import numpy_helper
import onnxruntime as ort 
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np 
import onnx 

from .base import SingleStageMethod


class EigenPlacesD128(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D128",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=128,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=128,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)


class EigenPlacesD256(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D256",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=256,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=256,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)


class EigenPlacesD512(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D512",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=512,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=512,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)


class EigenPlacesD2048(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D2048",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=2048,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)





class EigenPlacesD2048(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D2048",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=2048,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)




def quantize_eigenplaces_model(model: nn.Module, model_name: str): 
    model.eval() 
    dummy_input = torch.randn(1, 3, 512, 512)
    os.makedirs(os.path.join(os.path.dirname(__file__), "weights/EigenPlaces"), exist_ok=True)
    onnx_path = os.path.join(os.path.dirname(__file__), "weights/EigenPlaces", "tmp.onnx")
    torch.onnx.export(model,               # model being run
                 dummy_input,          # model input (or a tuple for multiple inputs)
                 onnx_path,            # where to save the model
                 export_params=True,   # store the trained parameter weights inside the model file
                 opset_version=11,     # the ONNX version to export the model to
                 do_constant_folding=True) 

    quantized_model_path = os.path.join(os.path.dirname(__file__), "weights/EigenPlaces", f"{model_name}.onnx")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )



class EigenPlacesINT8Model(nn.Module): 
    def __init__(self, model_name: str): 
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), "weights/EigenPlaces", f"{model_name}.onnx")
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
    
    


class EigenPlacesD2048INT8(SingleStageMethod):
    def __init__(
        self,
        name="EigenPlaces-D2048-INT8",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=2048,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
            quantize_eigenplaces_model(model, "EigenPlaces-D2048-INT8")
            model = EigenPlacesINT8Model("EigenPlaces-D2048-INT8")
        super().__init__(name, model, transform, descriptor_dim, search_dist)
