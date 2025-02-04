from datasets import Pitts30k, Essex3in1, Tokyo247, MSLS, SFXL
from methods import DinoV2_Salad, EigenPlaces, DinoV2_BoQ, ResNet50_BoQ
from metrics import ratk, extraction_latency, matching_latency, model_memory, database_memory

import config 
import csv
from tabulate import tabulate  # Add this import
import os 

methods = [DinoV2_Salad]#, EigenPlaces, DinoV2_BoQ, ResNet50_BoQ]
datasets = [(Pitts30k, config.Pitts30k_root)]
datasets = [(MSLS, config.MSLS_root), (Tokyo247, config.Tokyo247_root)] #(SFXL, config.SFXL_root)]

# Prepare data for CSV
csv_data = []
headers = ["Method", "Dataset", "Accuracy (R@1)", "Accuracy (R@5)", "Accuracy (R@10)", "Extraction Latency (ms)", "Matching Latency (ms)", "Model Memory (MB)", "DB Memory (MB)"]

for method_type in methods: 
    for (dataset_type, root) in datasets: 
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
            f"{database_memory(method, dataset):.2f}"
        ]
        csv_data.append(row)

# Print table
print("\nResults:")
print(tabulate(csv_data, headers=headers, tablefmt="grid"))

# Write to CSV file
output_file = 'data/results.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(csv_data)

print(f"\nResults have been saved to {output_file}")