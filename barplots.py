import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("summary_comp.csv")

df.set_index("transformation", inplace=True)

cols = [col for col in df.columns if "std" in col or "jaccard" in col]
print(cols)
df.drop(columns=cols, inplace=True)


df.columns = [col.replace("_mean_mrr", "") for col in df.columns]

df.rename(columns={"no_reranking": "none"}, inplace=True)

row_order = df.sort_values(by="none", ascending=False).index.tolist()
df = df.loc[row_order]

column_order = ["none"] + [col for col in df.columns if col != "none"]
df = df[column_order]

fig, ax = plt.subplots(figsize=(10, 8))

df.plot(kind="bar", ax=ax)

ax.set_title("Mean MRR for different transformations")
ax.set_xlabel("Transformation")
ax.set_ylabel("Score")
ax.legend(title="Reranking Method")

plt.tight_layout()

plt.savefig("barplots-bm25.png", dpi=300)

plt.show()
