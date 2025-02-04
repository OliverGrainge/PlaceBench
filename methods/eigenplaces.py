
import torch.nn as nn
import torch 
from .base import SingleStageMethod
import torchvision.transforms as T

class EigenPlaces(SingleStageMethod): 
    def __init__(
            self, 
            name="EigenPlaces",
            model=torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048),
        
            transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            descriptor_dim=2048, 
            search_dist='cosine'
   
    ): 
        super().__init__(name, model, transform, descriptor_dim, search_dist)



