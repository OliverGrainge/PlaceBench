import os 
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
from functools import partial

import os
from tabulate import tabulate  # Add this import
import gc 
import torch 

import config
from datasets import MSLS, SFXL, Essex3in1, Pitts30k, Tokyo247, SVOX
from methods import (
    CosPlacesD32,
    CosPlacesD64,
    CosPlacesD128,
    CosPlacesD512,
    CosPlacesD1024,
    CosPlacesD2048,
    CosPlacesD2048INT8,
    DinoV2_BoQ,
    DinoV2_BoQINT8,
    DinoV2_Salad,
    DinoV2_SaladINT8,
    EigenPlacesD128,
    EigenPlacesD256,
    EigenPlacesD512,
    EigenPlacesD2048,
    EigenPlacesD2048INT8,
    MixVPR,
    ResNet50_BoQ,
    ResNet50_BoQINT8,
    NetVLAD_SP,
    TeTRA,
)
from metrics import (
    database_memory,
    extraction_latency_cpu,
    extraction_latency_gpu,
    matching_latency,
    model_memory,
    ratk,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run a single VPR experiment')
    parser.add_argument('--method', type=str, required=True, help='Name of the method to run')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--metrics', type=str, nargs='+', required=True, help='List of metrics to compute')
    return parser.parse_args()



quant_ds = Essex3in1(config.Essex3in1_root)
# Method name mappings
METHOD_MAP = {
    'cosplaces32': CosPlacesD32,
    'cosplaces64': CosPlacesD64,
    'cosplaces128': CosPlacesD128,
    'cosplaces512': CosPlacesD512,
    'cosplaces1024': CosPlacesD1024,
    'cosplaces2048': CosPlacesD2048,
    'cosplaces2048int8': partial(CosPlacesD2048INT8, quant_ds=quant_ds),
    'eigenplaces128': EigenPlacesD128,
    'eigenplaces256': EigenPlacesD256,
    'eigenplaces512': EigenPlacesD512,
    'eigenplaces2048': EigenPlacesD2048,
    'eigenplaces2048int8': partial(EigenPlacesD2048INT8, quant_ds=quant_ds),
    'dinov2_boq': DinoV2_BoQ,
    'dinov2_salad': DinoV2_Salad,
    'dinov2_boqint8': partial(DinoV2_BoQINT8, quant_ds=quant_ds),
    'dinov2_saladint8': partial(DinoV2_SaladINT8, quant_ds=quant_ds),
    'resnet50_boq': ResNet50_BoQ,
    'resnet50_boqint8': partial(ResNet50_BoQINT8, quant_ds=quant_ds),
    'netvlad_sp': NetVLAD_SP,
    'tetra-boq-dd1': partial(TeTRA, aggregation_type="boq", descriptor_div=1),
    'tetra-boq-dd2': partial(TeTRA, aggregation_type="boq", descriptor_div=2),
    'tetra-gem-dd1': partial(TeTRA, aggregation_type="gem", descriptor_div=1),
    'tetra-gem-dd2': partial(TeTRA, aggregation_type="gem", descriptor_div=2),
    'tetra-salad-dd1': partial(TeTRA, aggregation_type="salad", descriptor_div=1),
    'tetra-salad-dd2': partial(TeTRA, aggregation_type="salad", descriptor_div=2),
    'tetra-mixvpr-dd1': partial(TeTRA, aggregation_type="mixvpr", descriptor_div=1),
    'tetra-mixvpr-dd2': partial(TeTRA, aggregation_type="mixvpr", descriptor_div=2),
}

# Dataset name mappings
DATASET_MAP = {
    'msls': partial(MSLS, root=config.MSLS_root),
    'essex3in1': partial(Essex3in1, root=config.Essex3in1_root),
    'pitts30k': partial(Pitts30k, root=config.Pitts30k_root),
    'tokyo247': partial(Tokyo247, root=config.Tokyo247_root),
    'svox-sun': partial(SVOX, root=config.SVOX_root, condition="sun"),
    'svox-rain': partial(SVOX, root=config.SVOX_root, condition="rain"),
    'svox-night': partial(SVOX, root=config.SVOX_root, condition="night"),
    'svox-snow': partial(SVOX, root=config.SVOX_root, condition="snow"),
}

# Metric name mappings
METRIC_MAP = {
    'r1': 'Accuracy (R@1)',
    'r5': 'Accuracy (R@5)',
    'r10': 'Accuracy (R@10)',
    'cpu_time': 'Extraction Latency CPU (ms)',
    'gpu_time': 'Extraction Latency GPU (ms)',
    'match_time': 'Matching Latency (ms)',
    'model_mem': 'Model Memory (MB)',
    'db_mem': 'DB Memory (MB)',
}


def main():
    args = parse_args()
    
    # Validate inputs
    if args.method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {args.method}. Available methods: {list(METHOD_MAP.keys())}")
    if args.dataset not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available datasets: {list(DATASET_MAP.keys())}")
    for metric in args.metrics:
        if metric not in METRIC_MAP:
            raise ValueError(f"Unknown metric: {metric}. Available metrics: {list(METRIC_MAP.keys())}")

    # Setup method and dataset
    method_class = METHOD_MAP[args.method]
    dataset_class = DATASET_MAP[args.dataset]
    
    # Handle quantization dataset if needed
    if args.method.endswith('INT8'):
        from datasets import Essex3in1
        import config
        quant_ds = Essex3in1(config.Essex3in1_root)
        method = partial(method_class, quant_ds=quant_ds)()
    else:
        method = method_class()

    dataset = dataset_class()

    # Map friendly metric names to actual metric names
    metrics_to_compute = [METRIC_MAP[m] for m in args.metrics]

    output_file = "data/results.csv"

    csv_data = []
    headers = [
        "Method",
        "Dataset",
        "Accuracy (R@1)",
        "Accuracy (R@5)",
        "Accuracy (R@10)",
        "Extraction Latency CPU (ms)",
        "Extraction Latency GPU (ms)",
        "Matching Latency (ms)",
        "Model Memory (MB)",
        "DB Memory (MB)",
    ]

    row = [method.name, dataset.name] + [""] * (len(headers) - 2)

    print(f"\nRunning experiment: {method.name} on {dataset.name}")
    print("-" * 50)

    # Compute only requested metrics
    if any(metric.startswith("Accuracy") for metric in metrics_to_compute):
        features =method.compute_features(dataset, batch_size=12, num_workers=0)
        recalls = ratk(method, dataset, topks=[1, 5, 10])
        del features  # Explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
        if "Accuracy (R@1)" in metrics_to_compute:
            row[headers.index("Accuracy (R@1)")] = f"{recalls[0]:.2f}"
            print(f"R@1: {recalls[0]:.2f}")
        if "Accuracy (R@5)" in metrics_to_compute:
            row[headers.index("Accuracy (R@5)")] = f"{recalls[1]:.2f}"
            print(f"R@5: {recalls[1]:.2f}")
        if "Accuracy (R@10)" in metrics_to_compute:
            row[headers.index("Accuracy (R@10)")] = f"{recalls[2]:.2f}"
            print(f"R@10: {recalls[2]:.2f}")

    if "Extraction Latency CPU (ms)" in metrics_to_compute:
        latency = extraction_latency_cpu(method, dataset)
        row[headers.index("Extraction Latency CPU (ms)")] = f"{latency:.2f}"
        print(f"Extraction Latency CPU: {latency:.2f} ms")
    
    if "Extraction Latency GPU (ms)" in metrics_to_compute:
        latency = extraction_latency_gpu(method, dataset)
        row[headers.index("Extraction Latency GPU (ms)")] = f"{latency:.2f}"
        print(f"Extraction Latency GPU: {latency:.2f} ms")
    
    if "Matching Latency (ms)" in metrics_to_compute:
        latency = matching_latency(method, dataset)
        row[headers.index("Matching Latency (ms)")] = f"{latency:.2f}"
        print(f"Matching Latency: {latency:.2f} ms")
    
    if "Model Memory (MB)" in metrics_to_compute:
        memory = model_memory(method, dataset)
        row[headers.index("Model Memory (MB)")] = f"{memory:.2f}"
        print(f"Model Memory: {memory:.2f} MB")
    
    if "DB Memory (MB)" in metrics_to_compute:
        memory = database_memory(method, dataset)
        row[headers.index("DB Memory (MB)")] = f"{memory:.2f}"
        print(f"Database Memory: {memory:.2f} MB")

    # Read current CSV data
    existing_data = {}
    if os.path.exists(output_file):
        with open(output_file, "r", newline="") as f:
            reader = csv.reader(f)
            file_headers = next(reader)  # Skip header row
            for csv_row in reader:
                if not csv_row:  # Skip empty rows
                    continue
                key = (csv_row[0], csv_row[1])
                existing_data[key] = {headers[i]: csv_row[i] for i in range(len(headers))}

    # Update or add new result
    key = (row[0], row[1])
    if key not in existing_data:
        existing_data[key] = {headers[i]: row[i] for i in range(len(headers))}
    else:
        for i, value in enumerate(row):
            if value:  # Only update if the value is not empty
                existing_data[key][headers[i]] = value

    # Convert to list format and write to CSV
    final_data = []
    for data_key in existing_data:
        csv_row = [existing_data[data_key][header] for header in headers]
        final_data.append(csv_row)

    # Write updated results to CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(final_data)

    print(f"Results updated in {output_file}")
    print("-" * 50)
    del method 
    del dataset

    # Print final table
    print("\nFinal Results:")
    print(tabulate(final_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
