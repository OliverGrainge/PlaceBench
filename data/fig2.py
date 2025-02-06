import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# Create the figure with two subplots sharing y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(7, 4))
fig.subplots_adjust(wspace=0.05)  # Reduce space between subplots

# Create scatter plots by model type
for idx, row in df.iterrows():
    if "GeM" in row['Method'] or "MixVPR" == row['Method'] or "ResNet50-BoQ" in row['Method']:
        continue
    
    # Determine which axis to use based on latency value
    latency = row["Matching Latency (ms)"]
    ax = ax1 if latency <= 35 else ax2
    
    if any(model in row['Method'] for model in ['CosPlaces', 'EigenPlaces', 'TeTRA', 'DinoV2']):
        for model, style in model_styles.items():
            if model in row['Method']:
                print(row["Method"])
                ax.scatter(
                    latency,
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
        print("=====", row["Method"])
        ax.scatter(
            latency,
            row["Accuracy (R@1)"],
            alpha=0.8,
            s=100,
            c='#808080',  # Gray color for other methods
            marker='o',
            edgecolor='white',
            linewidth=1,
            label=row['Method']  # Use the actual method name as label
        )

# Set the limits for each axis
ax1.set_xlim(-2, 35)
ax2.set_xlim(75, 200)

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
    ax.set_xlabel("")  # Remove individual x-labels

# Add labels and title
ax1.set_ylabel("Accuracy R@1 (%)")
ax1.set_xlabel("")  # Remove individual x-label
ax2.set_xlabel("")  # Remove individual x-label
plt.suptitle(f"{DATASET} R@1 vs. Matching Latency", y=0.98)

# Add a single x-label with the same style as y-label (moved up slightly)
fig.text(0.5, 0.04, "Matching Latency (ms)", ha='center', va='center', fontsize=11, fontfamily='serif')

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

# First adjust the main layout
plt.tight_layout()

# Then adjust the bottom spacing for the x-label while preserving the top spacing
plt.subplots_adjust(bottom=0.15, top=0.92)

# After layout adjustment, add the broken axis indicators again
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d), (1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d,+d), (-d,+d), **kwargs)
ax2.plot((-d,+d), (1-d,1+d), **kwargs)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/scatter_performance_vs_latency.jpg", dpi=300, bbox_inches='tight')
plt.show()
