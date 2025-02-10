import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from optimum.quanto import quantize, qint8, qint4
from optimum.quanto import freeze
from optimum.quanto import Calibration



from .base import SingleStageMethod

def calibrate(model, data_loader): 
    with Calibration(momentum=0.8):
        for idx, (image, _) in tqdm(enumerate(data_loader), total=5, desc="Calibrating"):
            if idx > 5:
                break
            model(image.to(next(model.parameters()).device))


class ResNet50_BoQ(SingleStageMethod):
    def __init__(
        self,
        name="ResNet50-BoQ",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (384, 384),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=16384,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "amaralibey/bag-of-queries",
                "get_trained_boq",
                backbone_name="resnet50",
                output_dim=16384,
            )
        super().__init__(name, model, transform, descriptor_dim, search_dist)

    def forward(self, input: Union[Image.Image, torch.Tensor]) -> dict:
        if isinstance(input, Image.Image):
            input = self.transform(input)[None, ...]
        return {
            "global_desc": self.model(input)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        }



class ResNet50_BoQINT8(SingleStageMethod):
    def __init__(
        self,
        name="ResNet50-BoQ-INT8",
        model=None,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (384, 384),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=16384,
        search_dist="cosine",
        quant_ds=None,
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "amaralibey/bag-of-queries",
                "get_trained_boq",
                backbone_name="resnet50",
                output_dim=16384,
            )
        dataloader = quant_ds.dataloader(batch_size=2, num_workers=0, transform=transform)
        quantize(model, weights=qint8, activations=qint8)
        calibrate(model, dataloader)
        freeze(model)
        super().__init__(name, model, transform, descriptor_dim, search_dist)

    def forward(self, input: Union[Image.Image, torch.Tensor]) -> dict:
        if isinstance(input, Image.Image):
            input = self.transform(input)[None, ...]
        return {
            "global_desc": self.model(input)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        }
    
    def compute_features(self, dataset, batch_size=1, num_workers=0, recompute=False, device=None):
        return super().compute_features(dataset, 1, 0, recompute, device)
