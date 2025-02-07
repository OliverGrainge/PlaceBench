import csv
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
    extraction_latency,
    matching_latency,
    model_memory,
    ratk,
)

output_file = "data/results.csv"
# methods = [DinoV2_Salad, EigenPlaces, DinoV2_BoQ, ResNet50_BoQ, MixVPR, TeTRA]
methods = [
    lambda: TeTRA(descriptor_div=1, aggregation_type="gem"),  # different aggregation
    lambda: TeTRA(descriptor_div=2, aggregation_type="gem"),  # different aggregation

    lambda: TeTRA(descriptor_div=1, aggregation_type="boq"),  # different aggregation
    lambda: TeTRA(descriptor_div=2, aggregation_type="boq"),  # different aggregation

    lambda: TeTRA(descriptor_div=1, aggregation_type="salad"),  # different aggregation
    lambda: TeTRA(descriptor_div=2, aggregation_type="salad"),  # different aggregation

    lambda: TeTRA(descriptor_div=1, aggregation_type="mixvpr"),  # different aggregation
    lambda: TeTRA(descriptor_div=2, aggregation_type="mixvpr"),  # different aggregation

    ResNet50_BoQ,
    DinoV2_BoQ, 
    DinoV2_Salad, 
    MixVPR,
    CosPlacesD32,
    CosPlacesD64,
    CosPlacesD128,
    CosPlacesD512,
    CosPlacesD1024,
    CosPlacesD2048,
    EigenPlacesD128,
    EigenPlacesD256,
    EigenPlacesD512,
    EigenPlacesD2048,
    
]

# Define which metrics to run
# Format: (method, dataset, [metrics_to_compute])
experiments_to_run = [
    # Example configurations:
    (EigenPlacesD512, Essex3in1, ["Extraction Latency (ms)"]),


]

csv_data = []
headers = [
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

for method_type, dataset_type, metrics_to_compute in experiments_to_run:
    # Get dataset root from config based on dataset type
    root = getattr(config, f"{dataset_type.__name__}_root")
    print("loading dataset")
    dataset = dataset_type(root)
    print("loading method")
    method = method_type() if callable(method_type) else method_type
    print("loaded")

    # Initialize row with method and dataset
    row = [method.name, dataset.name] + [""] * (len(headers) - 2)

    # Compute only requested metrics
    if any(metric.startswith("Accuracy") for metric in metrics_to_compute):
        method.compute_features(dataset, batch_size=12, num_workers=0)
        recalls = ratk(method, dataset, topks=[1, 5, 10])
        if "Accuracy (R@1)" in metrics_to_compute:
            row[headers.index("Accuracy (R@1)")] = f"{recalls[0]:.2f}"
        if "Accuracy (R@5)" in metrics_to_compute:
            row[headers.index("Accuracy (R@5)")] = f"{recalls[1]:.2f}"
        if "Accuracy (R@10)" in metrics_to_compute:
            row[headers.index("Accuracy (R@10)")] = f"{recalls[2]:.2f}"

    if "Extraction Latency (ms)" in metrics_to_compute:
        row[headers.index("Extraction Latency (ms)")] = f"{extraction_latency(method, dataset):.2f}"
    
    if "Matching Latency (ms)" in metrics_to_compute:
        row[headers.index("Matching Latency (ms)")] = f"{matching_latency(method, dataset):.2f}"
    
    if "Model Memory (MB)" in metrics_to_compute:
        row[headers.index("Model Memory (MB)")] = f"{model_memory(method, dataset):.2f}"
    
    if "DB Memory (MB)" in metrics_to_compute:
        row[headers.index("DB Memory (MB)")] = f"{database_memory(method, dataset):.2f}"

    csv_data.append(row)

# Read existing CSV data if it exists
existing_data = {}
if os.path.exists(output_file):
    with open(output_file, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header row
        for row in reader:
            # Skip empty rows
            if not row:
                continue
            # Use Method and Dataset as key
            key = (row[0], row[1])
            # Store each metric in a dictionary
            existing_data[key] = {headers[i]: row[i] for i in range(len(headers))}

# Update existing data with new results
for row in csv_data:
    key = (row[0], row[1])  # Method and Dataset
    if key not in existing_data:
        # Create new entry with all metrics
        existing_data[key] = {headers[i]: row[i] for i in range(len(headers))}
    else:
        # Update only non-empty values
        for i, value in enumerate(row):
            if value:  # Only update if the value is not empty
                existing_data[key][headers[i]] = value

# Convert back to list format for writing and display
final_data = []
for key in existing_data:
    row = [existing_data[key][header] for header in headers]
    final_data.append(row)

# Print updated table
print("\nResults:")
print(tabulate(final_data, headers=headers, tablefmt="grid"))

# Write updated results to CSV file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(final_data)

print(f"\nResults have been saved to {output_file}")
