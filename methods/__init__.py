from .base import SingleStageMethod
from .cosplaces import (
    CosPlacesD32,
    CosPlacesD64,
    CosPlacesD128,
    CosPlacesD256,
    CosPlacesD512,
    CosPlacesD1024,
    CosPlacesD2048,
    CosPlacesD2048INT8,
)
from .dinov2_boq import DinoV2_BoQ, DinoV2_BoQINT8
from .dinov2_salad import DinoV2_Salad, DinoV2_SaladINT8
from .eigenplaces import (
    EigenPlacesD128,
    EigenPlacesD256,
    EigenPlacesD512,
    EigenPlacesD2048,
    EigenPlacesD2048INT8,
)
from .mixvpr import MixVPR
from .resnet50_boq import ResNet50_BoQ, ResNet50_BoQINT8
from .netvlad_sp import NetVLAD_SP
from .tetra import TeTRA
