import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

# --- Unified Style for IROS ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,  # Disable LaTeX rendering if not needed
    }
)

DATASET = "Tokyo247"
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

# Define a consistent color for the scatter points
scatter_color = "#2F5597"  # Dark blue

# Define markers and colors for specific models
model_styles = {
    'CosPlaces': {'marker': 's', 'label': 'CosPlaces', 'color': '#2F5597'},  # Dark blue square
    'TeTRA': {'marker': '^', 'label': 'TeTRA', 'color': '#C00000'},          # Red triangle
    'EigenPlaces': {'marker': 'D', 'label': 'EigenPlaces', 'color': '#548235'}, # Green diamond
    'DinoV2': {'marker': 'P', 'label': 'DinoV2', 'color': '#7030A0'}             # Purple plus
}

# Default style for other models
default_style = {'marker': 'o', 'label': 'Other Models', 'color': '#808080'}  # Gray

# Create the figure with two subplots sharing y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(7, 4))
fig.subplots_adjust(wspace=0.05)  # Reduce space between subplots

# Create scatter plots by model type
for idx, row in df.iterrows():
    # Skip methods containing "GeM"
    if "GeM" in row['Method'] or "ResNet50-BoQ" in row['Method'] or "MixVPR" in row['Method']:
        continue
    
    # Determine which axis to use based on memory value
    memory = row["DB Memory (MB)"]
    ax = ax1 if memory <= 150 else ax2
    
    if any(model in row['Method'] for model in ['CosPlaces', 'EigenPlaces', 'TeTRA', 'DinoV2']):
        for model, style in model_styles.items():
            if model in row['Method']:
                ax.scatter(
                    memory,
                    row["Accuracy (R@1)"],
                    alpha=0.8,
                    s=100,
                    c=style['color'],
                    marker=style['marker'],
                    edgecolor='white',
                    linewidth=1,
                    label=style['label']
                )
                break
    else:
        ax.scatter(
            memory,
            row["Accuracy (R@1)"],
            alpha=0.8,
            s=100,
            c='#808080',  # Gray color for other methods
            marker='o',
            edgecolor='white',
            linewidth=1,
            label='Other Models'
        )

# Set the limits for each axis
ax1.set_xlim(-10, 200)
ax2.set_xlim(3500, 5000)

# Add broken axis indicators
d = .015  # size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d,+d), (-d,+d), **kwargs)
ax2.plot((-d,+d), (1-d,1+d), **kwargs)

# Customize both axes
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")

# Add labels and title
ax1.set_ylabel("Accuracy R@1 (%)")
plt.suptitle(f"{DATASET} R@1 vs. Database Memory", y=0.98)

# Add a single x-label
fig.text(0.5, 0.04, "Database Memory (MB)", ha='center', va='center', fontsize=11, fontfamily='serif')

# Add legend to the first axis only (with duplicate labels removed)
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles + handles2
all_labels = labels + labels2
by_label = dict(zip(all_labels, all_handles))
ax1.legend(by_label.values(), by_label.keys(), 
          loc='lower right', 
          fontsize=8, 
          frameon=True)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.92)

# Redraw broken axis indicators after layout adjustment
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d,+d), (-d,+d), **kwargs)
ax2.plot((-d,+d), (1-d,1+d), **kwargs)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig1.jpg", dpi=300, bbox_inches='tight')
plt.show()
