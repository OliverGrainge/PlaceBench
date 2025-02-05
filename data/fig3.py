import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

# --- Unified Style for IROS ---
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
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

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
plt.savefig("figures/bar_memory_consumption.jpg", dpi=300, bbox_inches='tight')
plt.show()
