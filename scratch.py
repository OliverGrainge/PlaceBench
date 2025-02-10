from methods import NetVLAD_SP 
import config 
from datasets import Pitts30k, MSLS, SFXL, Essex3in1, Tokyo247, Essex3in1, SVOX
from metrics import ratk, model_memory
import torch 
from methods import ResNet50_BoQ, DinoV2_BoQ

from methods import CosPlacesD2048INT8, CosPlacesD2048

ds = Pitts30k(config.Pitts30k_root)
model = CosPlacesD2048INT8(quant_ds=ds)

print(f"CosPlaces-D2048-INT8 R@1: {ratk(model, ds)}")

model = CosPlacesD2048()
print(f"CosPlaces-D2048R@1: {ratk(model, ds)}")


"""
model = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )

model = DinoV2_BoQ()
model = model.to("cuda")
from tqdm import tqdm
from optimum.quanto import quantize, qint8
from optimum.quanto import freeze
from optimum.quanto import Calibration

ds = Pitts30k(config.Pitts30k_root)

def calibrate(model, data_loader): 
    with Calibration(momentum=0.95):
        for idx, (image, _) in tqdm(enumerate(data_loader), total=25):
            if idx > 25:
                break
            model(image.to(next(model.parameters()).device))

calibrate(model, ds.dataloader(transform=model.transform))

quantize(model, weights=qint8, activations=qint8)

freeze(model)
"""