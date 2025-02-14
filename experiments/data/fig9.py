import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tabulate import tabulate

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.usetex": False,
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
    "TeTRA-BoQ-DD[2]",
]

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

df = df[df["Method"].isin(METHODS_TO_INCLUDE)]
df["Method"] = df["Method"].map(METHOD_NAMES_MAP)
df["Method"] = pd.Categorical(
    df["Method"], 
    categories=[METHOD_NAMES_MAP[m] for m in METHODS_TO_INCLUDE], 
    ordered=True
)
df = df.sort_values("Method")

# Basic safety checks
if df["DB Memory (MB)"].isna().any() or (df["DB Memory (MB)"] <= 0).any():
    raise ValueError("DB Memory column contains missing or invalid values")
if df["Matching Latency (ms)"].isna().any() or (df["Matching Latency (ms)"] <= 0).any():
    raise ValueError("Matching Latency column contains missing or invalid values")

methods = df["Method"].tolist()
x = np.arange(len(methods))  # x positions for each method
bar_width = 0.3
offset = 0.17  # This creates the gap between memory and latency bars

# Colors
color_model_mem = "#ED7D31"   # for Model Memory
color_db_mem    = "#2F5597"   # for DB Memory
color_extraction = "#ED7D31"  # for Extraction Latency
color_matching   = "#2F5597"  # for Matching Latency

fig, ax = plt.subplots(figsize=(8,6))
ax2 = ax.twinx()  # second y-axis for latency

# Turn on grid only for left axis (memory)
ax.grid(True, axis='y', linestyle='-', alpha=0.2)
ax2.grid(False)

# Stacked bars for Memory (on ax)
# Move them further to the left
mem_bottom = ax.bar(
    x - offset, 
    df["Model Memory (MB)"], 
    width=bar_width,
    color=color_model_mem,
    label="Model Memory (MB)"
)
mem_top = ax.bar(
    x - offset,
    df["DB Memory (MB)"],
    width=bar_width,
    bottom=df["Model Memory (MB)"],
    color=color_db_mem,
    label="DB Memory (MB)"
)

# Stacked bars for Latency (on ax2)
# Move them further to the right
lat_bottom = ax2.bar(
    x + offset,
    df["Extraction Latency GPU (ms)"],
    width=bar_width,
    color=color_extraction,
    label="Extraction Latency (ms)"
)
lat_top = ax2.bar(
    x + offset,
    df["Matching Latency (ms)"],
    width=bar_width,
    bottom=df["Extraction Latency GPU (ms)"],
    color=color_matching,
    label="Matching Latency (ms)"
)

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_ylabel("Memory (MB)")
ax2.set_ylabel("Latency (ms)")
ax.set_title(f"Memory and Latency Comparison on {DATASET}")

# Combine legends manually:
mem_legend = ax.legend(
    [mem_bottom, mem_top], 
    ["Model Memory (MB)", "DB Memory (MB)"],
    loc="upper left",
    fontsize=10,
    bbox_to_anchor=(0,1.15)
)
lat_legend = ax2.legend(
    [lat_bottom, lat_top], 
    ["Extraction Latency (ms)", "Matching Latency (ms)"],
    loc="upper right",
    fontsize=10,
    bbox_to_anchor=(1,1.15)
)
ax.add_artist(mem_legend)  # ensure the first legend remains visible

# Tidy up spines
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_linewidth(0.5)
ax2.spines["top"].set_linewidth(0.5)
ax2.spines["bottom"].set_linewidth(0.5)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/combined_memory_latency.jpg", dpi=300, bbox_inches="tight")
plt.show()
