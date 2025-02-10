import pandas as pd

df = pd.read_csv("results.csv")

columns = ["MSLS", "Pitts30k", "Tokyo247"]
methods = ["DinoV2-Salad", "DinoV2-BoQ", "ResNet50-BoQ", "EigenPlaces-D2048", 
          "CosPlaces-D2048", "TeTRA-BoQ-DD[1]", "TeTRA-BoQ-DD[2]", 
          "TeTRA-SALAD-DD[1]", "TeTRA-SALAD-DD[2]"]

pivot_df_memory = df.pivot(
    index='Method', 
    columns='Dataset', 
    values='DB Memory (MB)'
)


# Select only the relevant columns and create a pivot table
pivot_df = df.pivot(
    index='Method',
    columns='Dataset',
    values='Accuracy (R@1)'
)

filtered_df = pivot_df.loc[methods, columns]

filtered_df_memory = pivot_df_memory.loc[methods, columns]

filtered_df_memory_efficieny = filtered_df / filtered_df_memory

filtered_df_memory_efficieny = filtered_df_memory_efficieny.rename(columns={
    "MSLS": "MSLS/MB",
    "Pitts30K": "Pitts30K/MB",
    "Tokyo247": "Tokyo247/MB"
})

df = pd.concat([filtered_df, filtered_df_memory_efficieny], axis=1)

# Convert to LaTeX table with formatting
latex_table = df.to_latex(
    float_format=lambda x: "{:.2f}".format(x),
    caption="R@1 Accuracy comparison across datasets",
    label="tab:accuracy_comparison",
    escape=False,
)

# Modify the table environment to span two columns
latex_table = latex_table.replace(
    '\\begin{table}',
    '\\begin{table*}'
).replace(
    '\\end{table}',
    '\\end{table*}'
)

print(latex_table)
