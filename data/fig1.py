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

DATASET = "MSLS"
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

# Define a consistent color for the scatter points
scatter_color = "#2F5597"  # Dark blue

# Define markers and colors for specific models
model_styles = {
    'CosPlaces': {'marker': 's', 'label': 'CosPlaces', 'color': '#2F5597'},  # Dark blue square
    'TeTRA': {'marker': '^', 'label': 'TeTRA', 'color': '#C00000'},          # Red triangle
    'EigenPlaces': {'marker': 'D', 'label': 'EigenPlaces', 'color': '#548235'}, # Green diamond
    'DINO': {'marker': 'P', 'label': 'DINO', 'color': '#7030A0'}             # Purple plus
}

# Default style for other models
default_style = {'marker': 'o', 'label': 'Other Models', 'color': '#808080'}  # Gray

# Create the figure
plt.figure(figsize=(7, 4))

# Create scatter plots by model type
for idx, row in df.iterrows():
    if any(model in row['Method'] for model in ['CosPlaces', 'EigenPlaces', 'TeTRA', 'DINO']):
        for model, style in model_styles.items():
            if model in row['Method']:
                plt.scatter(
                    row["DB Memory (MB)"],
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
        plt.scatter(
            row["DB Memory (MB)"],
            row["Accuracy (R@1)"],
            alpha=0.8,
            s=100,
            c=default_style['color'],
            marker=default_style['marker'],
            edgecolor='white',
            linewidth=1,
            label=default_style['label']
        )

# Add legend (with duplicate labels removed)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
          loc='best', 
          fontsize=8, 
          frameon=True)

# Axis labels and title
plt.xlabel("Database Memory (MB)")
plt.ylabel("Accuracy R@1 (%)")
plt.title(f"Model Performance vs. Database Memory Usage on {DATASET}")

# Customize spines
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

# Adjust layout and save
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/scatter_performance_vs_db_memory.jpg", dpi=300, bbox_inches='tight')
plt.show()
