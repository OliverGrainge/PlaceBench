import os
from contextlib import redirect_stderr, redirect_stdout

import torch
import torchvision.transforms as T

from .base import SingleStageMethod


class MixVPR(SingleStageMethod):
    def __init__(
        self,
        name="MixVPR",
        model=None,
        transform=T.Compose(
            [
                T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=4096,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "jarvisyjw/MixVPR", "get_trained_model", pretrained=True
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)
