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


datasets = [(Tokyo247, config.Tokyo247_root), (MSLS, config.MSLS_root),(Pitts30k, config.Pitts30k_root)]
methods = [NetVLAD_SP]
datasets = [(Pitts30k, config.Pitts30k_root)]
# Prepare data for CSV
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

for dataset_type, root in datasets:
    for method_type in methods:
        #try:
        dataset = dataset_type(root)
        # Handle both regular classes and lambda functions
        method = method_type() if callable(method_type) else method_type
        method.compute_features(dataset, batch_size=12, num_workers=0)

        recalls = ratk(method, dataset, topks=[1, 5, 10])
        row = [
            method.name,
            dataset.name,
            f"{recalls[0]:.2f}",
            f"{recalls[1]:.2f}",
            f"{recalls[2]:.2f}",
            f"{extraction_latency(method, dataset):.2f}",
            f"{matching_latency(method, dataset):.2f}",
            f"{model_memory(method, dataset):.2f}",
            f"{database_memory(method, dataset):.2f}",
        ]
        csv_data.append(row)
        #except Exception as e:
        #    print(f"Error processing method")
        #    continue


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
            existing_data[key] = row

# Update existing data with new results
for row in csv_data:
    key = (row[0], row[1])  # Method and Dataset
    existing_data[key] = row

# Convert back to list format for writing and display
final_data = list(existing_data.values())

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
