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

DATASET = "Pitts30k"
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
# Define colors

tetra_color = "#2F5597"  # Dark blue for TeTRA
other_color = "#C55A11"  # Dark orange for other methods


# Assign color based on method name
colors = [tetra_color if "TeTRA" in method else other_color for method in df["Method"]]
# Create figure
plt.figure(figsize=(7, 4))
# Plot bar chart
bars = plt.bar(df["Method"], df["DB Memory (MB)"], color=colors, width=0.6)

# Axis labels and title
plt.ylabel("Database Memory (MB)")
plt.xlabel("Method")
plt.title(f"Memory Consumption Comparison on {DATASET}")

# Rotate x-labels for readability
plt.xticks(rotation=45, ha="right")

# Annotate bars with Accuracy (R@1)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.5,
        f'R@1: {df["Accuracy (R@1)"].iloc[i]:.1f}%',
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
plt.savefig("figures/fig4.jpg", dpi=300, bbox_inches='tight')
plt.show()
