import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
# Create figure
plt.figure(figsize=(7, 4))

# Create stacked bar chart
bottom_bars = []
top_bars = []
for i, (method, data) in enumerate(df.iterrows()):
    bottom = plt.bar(data["Method"], data["Model Memory (MB)"], 
                    color=tetra_colors[1], width=0.6)
    top = plt.bar(data["Method"], data['DB Memory (MB)'], 
                  bottom=data["Model Memory (MB)"],
                  color=tetra_colors[0], width=0.6)
    bottom_bars.append(bottom)
    top_bars.append(top)

# Axis labels and title
plt.ylabel("Memory (MB)")
plt.xlabel("Method")
plt.title(f"Memory Consumption Comparison on {DATASET}")

# Rotate x-labels for readability
plt.xticks(rotation=45, ha="right")

# Add legend
plt.legend(["Model Memory", "Database Memory"], loc='upper right')

# Annotate bars with Accuracy (R@1)
for i, (method, method_data) in enumerate(df.iterrows()):
    print(type(method_data["DB Memory (MB)"]))
    total_height = method_data["DB Memory (MB)"] + method_data["Model Memory (MB)"]
    
    plt.text(
        i,
        total_height + 0.5,
        f'{method_data["Accuracy (R@1)"]:.1f}%',
        ha="center",
        va="bottom",
        fontsize=9,
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
plt.savefig("figures/fig5.jpg", dpi=300, bbox_inches='tight')
plt.show()
