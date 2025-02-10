import os 
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from experiments import experiments_to_run

import os
from tabulate import tabulate  # Add this import

import config
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
    extraction_latency_cpu,
    extraction_latency_gpu,
    matching_latency,
    model_memory,
    ratk,
)

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

for method_type, dataset_type, metrics_to_compute in experiments_to_run:
    method = method_type() 
    dataset = dataset_type()
    row = [method.name, dataset.name] + [""] * (len(headers) - 2)

    print(f"\nRunning experiment: {method.name} on {dataset.name}")
    print("-" * 50)

    # Compute only requested metrics
    if any(metric.startswith("Accuracy") for metric in metrics_to_compute):
        method.compute_features(dataset, batch_size=12, num_workers=0)
        recalls = ratk(method, dataset, topks=[1, 5, 10])
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

# Print final table
print("\nFinal Results:")
print(tabulate(final_data, headers=headers, tablefmt="grid"))
