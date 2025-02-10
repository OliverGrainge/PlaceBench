import os
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torchvision.transforms as T

from .base import SingleStageMethod

from tqdm import tqdm
from optimum.quanto import quantize, qint8, qint4
from optimum.quanto import freeze
from optimum.quanto import Calibration

def calibrate(model, data_loader): 
    with Calibration(momentum=0.8):
        for idx, (image, _) in tqdm(enumerate(data_loader), total=5, desc="Calibrating"):
            if idx > 5:
                break
            model(image.to(next(model.parameters()).device))


class DinoV2_Salad(SingleStageMethod):
    def __init__(
        self,
        name="DinoV2-Salad",
        model=None,  # We'll load the model below
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (322, 322),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=8448,
        search_dist="cosine",
    ):
        # Suppress output while loading model
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load("serizba/salad", "dinov2_salad")

        super().__init__(name, model, transform, descriptor_dim, search_dist)



class DinoV2_SaladINT8(SingleStageMethod):
    def __init__(
        self,
        name="DinoV2-Salad-INT8",
        model=None,  # We'll load the model below
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (322, 322),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=8448,
        search_dist="cosine",
        quant_ds=None,
    ):
        # Suppress output while loading model
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load("serizba/salad", "dinov2_salad")

        dataloader = quant_ds.dataloader(batch_size=2, num_workers=0, transform=transform)
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #model = model.to(device)
        quantize(model, weights=qint8, activations=qint8)
        calibrate(model, dataloader)
        freeze(model)

        super().__init__(name, model, transform, descriptor_dim, search_dist)
