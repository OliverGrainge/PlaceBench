import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
from tabulate import tabulate

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,
    }
)

DATASET = "Tokyo247"
METHODS_TO_INCLUDE = [
    "ResNet50-BoQ",
    "DinoV2-BoQ",
    "DinoV2-Salad",
    "CosPlaces-D2048",
    "EigenPlaces-D2048",
    "TeTRA-BoQ-DD[1]",
    "TeTRA-BoQ-DD[2]",
]

# Define mapping for method names
METHOD_NAMES_MAP = {
    "ResNet50-BoQ": "ResNet50-BoQ",
    "DinoV2-BoQ": "DinoV2-BoQ",
    "DinoV2-Salad": "DinoV2-Salad",
    "CosPlaces-D2048": "CosPlaces",
    "EigenPlaces-D2048": "EigenPlaces",
    "TeTRA-BoQ-DD[1]": "TeTRA-Salad",
    "TeTRA-BoQ-DD[2]": "TeTRA-BOQ",
}

df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()
# Filter dataframe to only include selected methods
df = df[df["Method"].isin(METHODS_TO_INCLUDE)]

# Map the method names to their new labels
df["Method"] = df["Method"].map(METHOD_NAMES_MAP)
# Set the Method column as categorical with the desired order
df["Method"] = pd.Categorical(
    df["Method"], 
    categories=[METHOD_NAMES_MAP[m] for m in METHODS_TO_INCLUDE], 
    ordered=True
)

# Sort the dataframe based on the categorical order
df = df.sort_values("Method")

# Define colors for stacked bars

# Define colors for stacked bars
tetra_colors = ["#2F5597", "#ED7D31"]  # Dark and light blue for TeTRA

if df['Matching Latency (ms)'].isna().any() or (df['Matching Latency (ms)'] <= 0).any():
    print("Warning: Missing or invalid values found in Matching Latency column:")
    print(df[['Method', 'Matching Latency (ms)']])
    raise ValueError("Matching Latency column contains missing or invalid values")

# Create figure
plt.figure(figsize=(7, 4))

# Create table of latency data
table_data = []
for i, (method, data) in enumerate(df.iterrows()):
    total_latency = data['Matching Latency (ms)'] + data['Extraction Latency GPU (ms)']
    table_data.append([
        data['Method'],
        data['Extraction Latency GPU (ms)'],
        data['Matching Latency (ms)'],
        total_latency,
        f"{data['Accuracy (R@1)']:.1f}%"
    ])

print("\nLatency and Accuracy Summary:")
print(tabulate(
    table_data,
    headers=['Method', 'Extraction (ms)', 'Matching (ms)', 'Total (ms)', 'Accuracy (R@1)'],
    tablefmt='grid',
    floatfmt='.2f'
))

# Create stacked bar chart
bottom_bars = []
top_bars = []
for i, (method, data) in enumerate(df.iterrows()):
    bottom = plt.bar(data["Method"], data["Extraction Latency GPU (ms)"], 
                    color=tetra_colors[1], width=0.6)
    top = plt.bar(data["Method"], data['Matching Latency (ms)'], 
                  bottom=data["Extraction Latency GPU (ms)"],
                  color=tetra_colors[0], width=0.6)
    bottom_bars.append(bottom)
    top_bars.append(top)

# Axis labels and title
plt.ylabel("Latency (ms)")
plt.xlabel("Method")
plt.title(f"Latency Comparison on {DATASET}", fontsize=13)

# Rotate x-labels for readability
plt.xticks(rotation=45, ha="right")

# Add legend
plt.legend(["Extraction Latency", "Matching Latency"], loc='upper right', fontsize=12)

# Annotate bars with Accuracy (R@1)
for i, (method, method_data) in enumerate(df.iterrows()):
    total_height = method_data["Matching Latency (ms)"] + method_data["Extraction Latency GPU (ms)"]
    
    plt.text(
        i,
        total_height + 0.5,
        f'{method_data["Accuracy (R@1)"]:.1f}%',
        ha="center",
        va="bottom",
        fontsize=11,
    )


# Customize spines
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

# Adjust layout and save
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig4.jpg", dpi=300, bbox_inches='tight')
plt.show()
