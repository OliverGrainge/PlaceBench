import pandas as pd 

df = pd.read_csv("results.csv")

# Convert to LaTeX table with some formatting
latex_table = df.to_latex(
    index=False,
    float_format=lambda x: '{:.2f}'.format(x),
    caption='Performance comparison of different methods',
    label='tab:performance',
    escape=False
)

print(latex_table)