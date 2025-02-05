import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv("results.csv")


print(df.head())

# Filter for Pitts30k dataset and calculate total memory
pitts_data = df[df['Dataset'] == 'Pitts30k'].copy()
pitts_data['Total Memory'] = pitts_data['Model Memory (MB)'] + pitts_data['DB Memory (MB)']

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(pitts_data['Total Memory'], pitts_data['Accuracy (R@1)'], alpha=0.6)

# Add labels for each point
for idx, row in pitts_data.iterrows():
    plt.annotate(row['Method'], 
                (row['Total Memory'], row['Accuracy (R@1)']),
                xytext=(5, 5), 
                textcoords='offset points')

# Customize the plot
plt.xlabel('Total Memory (MB)')
plt.ylabel('Accuracy R@1 (%)')
plt.title('Model Performance vs Memory Usage on Pitts30k')
plt.grid(True, linestyle='--', alpha=0.7)

# Add some padding to the axes
plt.margins(0.1)

plt.tight_layout()
plt.show()