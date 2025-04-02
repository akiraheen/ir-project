# %%
import pandas as pd

df = pd.read_csv("results/metrics.csv")

# %%

df = df.drop(
    columns=["iteration", "filename", "original_results", "transformed_results"]
)
# %%

df = df.groupby(["transformation"]).mean().round(2)
df = df.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
# %%

df.to_csv("results/summary.csv")

# %%
