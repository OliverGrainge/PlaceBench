import os 
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import MSLS, SFXL, Essex3in1, Pitts30k, Tokyo247, SVOX
import config 
from functools import partial

from methods import (
    CosPlacesD32,
    CosPlacesD64,
    CosPlacesD128,
    CosPlacesD512,
    CosPlacesD1024,
    CosPlacesD2048,
    DinoV2_BoQ,
    DinoV2_Salad,
    DinoV2_BoQINT8,
    DinoV2_SaladINT8,
    EigenPlacesD128,
    EigenPlacesD256,
    EigenPlacesD512,
    EigenPlacesD2048,
    EigenPlacesD2048INT8,
    MixVPR,
    ResNet50_BoQ,
    NetVLAD_SP,
    TeTRA,
)

from metrics import (
    database_memory,
    extraction_latency,
    matching_latency,
    model_memory,
    ratk,
)


LIST_OF_METRICS = [
    "Method",
    "Dataset",
    "Accuracy (R@1)",
    "Accuracy (R@5)",
    "Accuracy (R@10)",
    "Extraction Latency (ms)",
    "Matching Latency (ms)",
    "Model Memory (MB)",
    "DB Memory (MB)",
]

baseline_methods = [
    CosPlacesD32,
    CosPlacesD64,
    CosPlacesD128,
    CosPlacesD512,
    CosPlacesD1024,
    CosPlacesD2048,
    DinoV2_BoQ,
    DinoV2_Salad,
    EigenPlacesD128,
    EigenPlacesD256,
    EigenPlacesD512,
    EigenPlacesD2048,
    EigenPlacesD2048INT8,
    MixVPR,
    ResNet50_BoQ,
    NetVLAD_SP,
]

quant_ds = Essex3in1(config.Essex3in1_root)
experiments_to_run = [
    # Accuracy Experiments 
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(Essex3in1, root=config.Essex3in1_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(Pitts30k, root=config.Pitts30k_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(MSLS, root=config.MSLS_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(Tokyo247, root=config.Tokyo247_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="sun"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="rain"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="night"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_SaladINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="snow"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(Essex3in1, root=config.Essex3in1_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(Pitts30k, root=config.Pitts30k_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(MSLS, root=config.MSLS_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(Tokyo247, root=config.Tokyo247_root), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="sun"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="rain"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="night"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),
    (partial(DinoV2_BoQINT8, quant_ds=quant_ds), partial(SVOX, root=config.SVOX_root, condition="snow"), ["Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)"]),

    

]