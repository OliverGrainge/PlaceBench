import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

# --- Unified Style for IROS ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": False,
    }
)

DATASET = "Tokyo247"
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

# Define markers and colors for specific models
model_styles = {
    'CosPlaces': {'marker': 's', 'label': 'CosPlaces', 'color': '#2F5597'},
    'TeTRA': {'marker': '^', 'label': 'TeTRA', 'color': '#C00000'},
    'EigenPlaces': {'marker': 'D', 'label': 'EigenPlaces', 'color': '#548235'},
    'DinoV2': {'marker': 'P', 'label': 'DinoV2', 'color': '#7030A0'}
}

# Create the figure with two subplots (removing the split panels)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3)

# Function to plot scatter points
def plot_points(ax, x_value, y_value, method, style_dict):
    if any(model in method for model in model_styles.keys()):
        for model, style in model_styles.items():
            if model in method:
                ax.scatter(x_value, y_value, alpha=0.8, s=100,
                         c=style['color'], marker=style['marker'],
                         edgecolor='white', linewidth=1, label=style['label'])
                break
    else:
        ax.scatter(x_value, y_value, alpha=0.8, s=100,
                  c='#808080', marker='o', edgecolor='white',
                  linewidth=1, label='Other Models')

# Plot Memory vs Accuracy (top plot)
for idx, row in df.iterrows():
    if "GeM" in row['Method'] or "ResNet50-BoQ" in row['Method'] or "MixVPR" == row['Method'] or "INT8" in row['Method'] or "DinoV2" in row['Method']:
        continue
    
    memory = row["DB Memory (MB)"]
    plot_points(ax1, memory, row["Accuracy (R@1)"], row['Method'], model_styles)

# Plot Latency vs Accuracy (bottom plot)
for idx, row in df.iterrows():
    if "GeM" in row['Method'] or "ResNet50-BoQ" in row['Method'] or "MixVPR" == row['Method'] or "INT8" in row['Method'] or "DinoV2" in row['Method']:
        continue
    
    latency = row["Matching Latency (ms)"]
    plot_points(ax2, latency, row["Accuracy (R@1)"], row['Method'], model_styles)

# Set axis limits (now using single ranges)
#ax1.set_xlim(-10, 5000)
#ax2.set_xlim(-2, 200)

# Customize axes
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.grid(True, alpha=0.3)

# Add labels
ax1.set_ylabel("Accuracy R@1 (%)")
ax2.set_ylabel("Accuracy R@1 (%)")

# Add subtitles for each plot
ax1.set_title(f"{DATASET} R@1 vs. Database Memory", pad=10, fontsize=13)
ax2.set_title(f"{DATASET} R@1 vs. Matching Latency", pad=10, fontsize=13)

# Add x-labels
ax1.set_xlabel("Database Memory (MB)", fontsize=13)
ax2.set_xlabel("Matching Latency (ms)", fontsize=13)

# Add legend (only once)
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
          loc='center right', bbox_to_anchor=(0.98, 0.5),
          fontsize=11, frameon=True)

# Save the figure
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig5.jpg", dpi=300, bbox_inches='tight')
plt.show()
