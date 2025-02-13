import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Unified Style for IROS ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False,
})

DATASET = "Tokyo247"
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

# Filter out unwanted methods
mask = ~(df['Method'].str.contains("GeM") | 
         df['Method'].str.contains("ResNet50-BoQ") |
         (df['Method'] == "MixVPR") |
         df['Method'].str.contains("INT8") |
         df['Method'].str.contains("DinoV2"))
df = df[mask].copy()

# Define markers and colors for specific models
model_styles = {
    'CosPlaces': {'marker': 's', 'label': 'CosPlaces', 'color': '#2F5597'},
    'TeTRA': {'marker': '^', 'label': 'TeTRA', 'color': '#C00000'},
    'EigenPlaces': {'marker': 'D', 'label': 'EigenPlaces', 'color': '#548235'},
    'DinoV2': {'marker': 'P', 'label': 'DinoV2', 'color': '#7030A0'}
}

# Helper function to determine the style for a given method
def get_style(method):
    for key, style in model_styles.items():
        if key in method:
            return style
    return {'marker': 'o', 'label': method, 'color': '#808080'}

# Helper function to determine the base model name
def get_base_model(method):
    if "CosPlaces" in method:
        return "CosPlaces"
    elif "TeTRA" in method: 
        return "TeTRA"
    elif "EigenPlaces" in method:
        return "EigenPlaces"
    elif "DinoV2" in method:
        return "DinoV2"
    return method

# Create the figure with two subplots - wider aspect ratio for two columns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
fig.subplots_adjust(hspace=0.35)

# Add a single main title for both plots
fig.suptitle(f"Accuracy vs. Memory and Latency Trade-offs on {DATASET}", 
            fontsize=14, y=0.98)

# --- Plot 1: Database Memory vs Accuracy (Line Plot with Markers) ---
df['BaseModel'] = df['Method'].apply(get_base_model)
for base_model, group in df.groupby('BaseModel'):
    style = get_style(base_model)
    group = group.sort_values("DB Memory (MB)")
    ax1.plot(group["DB Memory (MB)"], group["Accuracy (R@1)"],
             marker=style['marker'], markersize=8, linewidth=2,
             color=style['color'], label=style['label'], alpha=0.9)

ax1.set_xlabel("Database Memory (MB)", fontsize=13)
ax1.set_ylabel("Accuracy R@1 (%)", fontsize=13)
ax1.grid(True, alpha=0.3)
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)

# --- Plot 2: Matching Latency vs Accuracy (Line Plot with Markers) ---
for base_model, group in df.groupby('BaseModel'):
    style = get_style(base_model)
    # Sort the group by "Matching Latency (ms)" for meaningful line connections
    group = group.sort_values("Matching Latency (ms)")
    ax2.plot(group["Matching Latency (ms)"], group["Accuracy (R@1)"],
             marker=style['marker'], markersize=8, linewidth=2,
             color=style['color'], label=style['label'], alpha=0.9)

ax2.set_xlabel("Matching Latency (ms)", fontsize=13)
ax2.set_ylabel("Accuracy R@1 (%)", fontsize=13)
ax2.grid(True, alpha=0.3)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

# Create a combined legend without duplicates
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            labels.append(label)
            handles.append(handle)

# Add legend to the bottom plot (ax2) instead
ax2.legend(handles, labels, 
          loc='lower right',  # Changed to lower right
          bbox_to_anchor=(0.98, 0.02),  # Adjusted x coordinate to 0.98 for right alignment
          fontsize=12,
          frameon=True,
          framealpha=1.0,
          edgecolor='black',
          borderpad=1)

# Adjust the subplot spacing to accommodate the legend
fig.subplots_adjust(hspace=0.4, right=0.95)  # Keep some right margin for the legend

# Save the figure
os.makedirs("figures", exist_ok=True)
#plt.savefig("figures/fig5_line.jpg", dpi=300, bbox_inches='tight')
plt.show()
