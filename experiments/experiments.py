import os 
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import MSLS, SFXL, Essex3in1, Pitts30k, Tokyo247

from methods import (
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



accuray_experiments = 

baseline_accuracy_experiments = []