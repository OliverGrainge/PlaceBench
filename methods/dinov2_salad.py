import os
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from .base import SingleStageMethod


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
