import csv
import os

from tabulate import tabulate  # Add this import

import config
from datasets import MSLS, SFXL, Essex3in1, Pitts30k, Tokyo247
from methods import DinoV2_BoQ, DinoV2_Salad, EigenPlaces, ResNet50_BoQ, MixVPR, TeTRA
from metrics import (database_memory, extraction_latency, matching_latency,
                     model_memory, ratk)

output_file = "data/results.csv"
#methods = [DinoV2_Salad, EigenPlaces, DinoV2_BoQ, ResNet50_BoQ, MixVPR]
methods = [TeTRA]
datasets = [(Essex3in1, config.Essex3in1_root)]

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

for method_type in methods:
    for dataset_type, root in datasets:
        dataset = dataset_type(root)
        method = method_type()
        method.compute_features(dataset, batch_size=128, num_workers=8)

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


# Read existing CSV data if it exists
existing_data = {}
if os.path.exists(output_file):
    with open(output_file, 'r', newline='') as f:
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
