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
tetra_colors = ["#2F5597", "#ED7D31"]  # Dark and light blue for TeTRA

if df['DB Memory (MB)'].isna().any() or (df['DB Memory (MB)'] <= 0).any():
    print("Warning: Missing or invalid values found in DB Memory column:")
    print(df[['Method', 'DB Memory (MB)']])
    raise ValueError("DB Memory column contains missing or invalid values")

# Create table of memory usage
table_data = []
for i, (method, data) in enumerate(df.iterrows()):
    total_height = data["DB Memory (MB)"] + data["Model Memory (MB)"]
    table_data.append([
        data['Method'],
        data['Model Memory (MB)'],
        data['DB Memory (MB)'],
        total_height
    ])

print("\nMemory Usage Summary:")
print(tabulate(
    table_data,
    headers=['Method', 'Model Memory (MB)', 'DB Memory (MB)', 'Total Memory (MB)'],
    tablefmt='grid',
    floatfmt='.1f'
))

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

# Create stacked bar chart
bottom_bars = []
top_bars = []
for i, (method, data) in enumerate(df.iterrows()):
    total_height = data["DB Memory (MB)"] + data["Model Memory (MB)"]
    print(data["Method"], total_height)
    bottom = ax1.bar(data["Method"], data["Model Memory (MB)"], 
                    color=tetra_colors[1], width=0.6)
    top = ax1.bar(data["Method"], data['DB Memory (MB)'], 
                  bottom=data["Model Memory (MB)"],
                  color=tetra_colors[0], width=0.6)
    bottom_bars.append(bottom)
    top_bars.append(top)

# Customize first subplot
ax1.set_ylabel("Memory (MB)")
ax1.set_xlabel("Method")
ax1.set_title(f"Memory Consumption Comparison on {DATASET}", fontsize=13)
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
ax1.legend(["Model Memory", "Database Memory"], loc='upper right', fontsize=12)

# Annotate bars with Accuracy (R@1)
for i, (method, method_data) in enumerate(df.iterrows()):
    total_height = method_data["DB Memory (MB)"] + method_data["Model Memory (MB)"]
    
    ax1.text(
        i,
        total_height + 0.5,
        f'{method_data["Accuracy (R@1)"]:.1f}%',
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Customize spines
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_linewidth(0.5)
ax1.spines["bottom"].set_linewidth(0.5)

if df['Matching Latency (ms)'].isna().any() or (df['Matching Latency (ms)'] <= 0).any():
    print("Warning: Missing or invalid values found in Matching Latency column:")
    print(df[['Method', 'Matching Latency (ms)']])
    raise ValueError("Matching Latency column contains missing or invalid values")

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
    bottom = ax2.bar(data["Method"], data["Extraction Latency GPU (ms)"], 
                    color=tetra_colors[1], width=0.6)
    top = ax2.bar(data["Method"], data['Matching Latency (ms)'], 
                  bottom=data["Extraction Latency GPU (ms)"],
                  color=tetra_colors[0], width=0.6)
    bottom_bars.append(bottom)
    top_bars.append(top)

# Customize second subplot
ax2.set_ylabel("Latency (ms)")
ax2.set_xlabel("Method")
ax2.set_title(f"Latency Comparison on {DATASET}", fontsize=13)
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
ax2.legend(["Extraction Latency", "Matching Latency"], loc='upper right', fontsize=12)

# Annotate bars with Accuracy (R@1)
for i, (method, method_data) in enumerate(df.iterrows()):
    total_height = method_data["Matching Latency (ms)"] + method_data["Extraction Latency GPU (ms)"]
    
    ax2.text(
        i,
        total_height + 0.5,
        f'{method_data["Accuracy (R@1)"]:.1f}%',
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Customize spines
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(0.5)
ax2.spines["bottom"].set_linewidth(0.5)

# Adjust layout and save
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig6.jpg", dpi=300, bbox_inches='tight')
plt.show()

