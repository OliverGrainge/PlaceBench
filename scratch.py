from methods import NetVLAD_SP 
import config 
from datasets import Pitts30k, MSLS, SFXL, Essex3in1, Tokyo247, Essex3in1, SVOX
from metrics import ratk, model_memory
import torch 
from methods import ResNet50_BoQ, DinoV2_BoQ

from methods import CosPlacesD2048INT8, DinoV2_BoQINT8, EigenPlacesD2048INT8, DinoV2_SaladINT8, ResNet50_BoQINT8

from tqdm import tqdm
from optimum.quanto import quantize, qint8, qint4
from optimum.quanto import freeze
from optimum.quanto import Calibration
import torchvision.transforms as T


ds = Essex3in1(root=config.Essex3in1_root)

model = ResNet50_BoQINT8(quant_ds=ds)

model.compute_features(ds)