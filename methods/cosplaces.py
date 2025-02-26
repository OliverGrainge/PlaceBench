import os
from contextlib import redirect_stderr, redirect_stdout

import torch
import torchvision.transforms as T

from .base import SingleStageMethod


class CosPlacesD32(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D32",
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
        descriptor_dim=32,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=32,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD64(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D64",
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
        descriptor_dim=64,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=64,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD128(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D128",
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
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=128,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD256(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D256",
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
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=256,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD512(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D512",
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
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=512,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD1024(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D1024",
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
        descriptor_dim=1024,
        search_dist="cosine",
    ):
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            model = torch.hub.load(
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=1024,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)


class CosPlacesD2048(SingleStageMethod):
    def __init__(
        self,
        name="CosPlaces-D2048",
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
                "gmberton/cosplace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )

        super().__init__(name, model, transform, descriptor_dim, search_dist)
