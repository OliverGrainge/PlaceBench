import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .base import SingleStageMethod


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
