import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tabulate import tabulate

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False,
    "axes.linewidth": 1.2,  # Increased line width
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.5,
    "text.color": "black",  # Ensure all text is black
    "axes.labelcolor": "black",  # Ensure axis labels are black
    "axes.edgecolor": "black",  # Ensure axis edges are black
    "xtick.color": "black",  # Ensure tick labels are black
    "ytick.color": "black",  # Ensure tick labels are black
})

# Turn off all grid
plt.rcParams['axes.grid'] = False

DATASET = "Tokyo247"
METHODS_TO_INCLUDE = [
    "ResNet50-BoQ",
    "DinoV2-BoQ",
    "DinoV2-Salad",
    "CosPlaces-D2048",
    "EigenPlaces-D2048",
    "TeTRA-BoQ-DD[1]",
    "TeTRA-SALAD-DD[1]",
    "TeTRA-MixVPR-DD[1]"
]

METHOD_NAMES_MAP = {
    "ResNet50-BoQ": "ResNet50-BoQ",
    "DinoV2-BoQ": "DinoV2-BoQ",
    "DinoV2-Salad": "DinoV2-Salad",
    "CosPlaces-D2048": "CosPlaces",
    "EigenPlaces-D2048": "EigenPlaces",
    "TeTRA-BoQ-DD[1]": "TeTRA-BoQ",
    "TeTRA-SALAD-DD[1]": "TeTRA-Salad",
    "TeTRA-MixVPR-DD[1]": "TeTRA-MixVPR",
}

df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

df = df[df["Method"].isin(METHODS_TO_INCLUDE)]
df["Method"] = df["Method"].map(METHOD_NAMES_MAP)
df["Method"] = pd.Categorical(
    df["Method"], 
    categories=[METHOD_NAMES_MAP[m] for m in METHODS_TO_INCLUDE], 
    ordered=True
)
df = df.sort_values("Method")
print(df)


# Sort by total memory, for instance
df_sorted = df.assign(TotalMemory = df["Model Memory (MB)"] + df["DB Memory (MB)"])
df_sorted = df_sorted.sort_values("TotalMemory")

x = np.arange(len(df_sorted))

fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()
ax3 = ax1.twinx()  # Create a third axis

# Offset the right spine for ax3 to prevent overlap with ax2
ax3.spines['right'].set_position(('outward', 60))

# Plot the existing lines and add the accuracy line
ax1.plot(x, df_sorted["TotalMemory"], 'o-', color="#ED7D31", label="Total Memory", linewidth=2.5)
ax2.plot(x, df_sorted["Extraction Latency GPU (ms)"] + df_sorted["Matching Latency (ms)"], 
         's--', color="#2F5597", label="Total Latency", linewidth=2.5)
ax3.plot(x, df_sorted["Accuracy (R@1)"], '^-', color="#70AD47", label="Accuracy (R@1)", linewidth=2.5)
ax3.set_ylim(0, 100)

ax1.set_xticks(x)
ax1.set_xticklabels(df_sorted["Method"], rotation=45, ha="right")
ax1.set_ylabel("Total Memory (MB)")
ax2.set_ylabel("Total Latency (ms)")
ax3.set_ylabel("Accuracy (R@1)")
ax1.set_title(f"Methods sorted by Memory on {DATASET}")
ax1.grid(True, axis='y', linestyle='--', alpha=0.6, color='gray')

# Combine legends for all three lines
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="lower right")

plt.tight_layout()
plt.savefig("figures/fig3.png", dpi=300)
plt.show()
