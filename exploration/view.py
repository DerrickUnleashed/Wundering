import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

df = pd.read_parquet("data/sample_full.parquet")

print(df.head(4))

print("Shape:", df.shape)
print(df.head())

feature_cols = [c for c in df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction']]
print("Number of features:", len(feature_cols))

plt.figure(figsize=(14, 6))
sample_cols = feature_cols#[:25] if len(feature_cols) > 25 else feature_cols
sns.boxplot(data=df[sample_cols])
plt.xticks(rotation=90)
plt.title("Feature Distribution (Boxplot)")
plt.tight_layout()
plt.savefig("visualisations/boxplot.png", dpi=300)
plt.show()


corr_subset = df[sample_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_subset, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Sampled Features)")
plt.savefig("visualisations/heatmap.png", dpi=300)
plt.show()


first_seq = df.seq_ix.unique()[0]
seq_df = df[df.seq_ix == first_seq]

plt.figure(figsize=(12, 5))
plt.plot(seq_df["step_in_seq"], seq_df[feature_cols[0]])
plt.title(f"Feature Trend Over Steps (seq_ix={first_seq})")
plt.xlabel("step_in_seq")
plt.ylabel(feature_cols[0])
plt.savefig("visualisations/featuretrend.png", dpi=300)
plt.show()

warmup = df[df.step_in_seq < 100][feature_cols].mean()
predict = df[df.step_in_seq >= 100][feature_cols].mean()

plt.figure(figsize=(14, 5))
plt.plot(warmup.values, label="Warm-up mean")
plt.plot(predict.values, label="Prediction mean")
plt.title("Warm-up vs Prediction Drift (Mean Values)")
plt.legend()
plt.savefig("visualisations/drift.png", dpi=300)
plt.show()

plt.figure(figsize=(14, 5))
sns.heatmap(df[feature_cols].isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.savefig("visualisations/missing.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 4))
autocorrelation_plot(seq_df[feature_cols[0]])
plt.title("Autocorrelation of First Feature (Single Sequence)")
plt.savefig("visualisations/autocorr.png", dpi=300)
plt.show()