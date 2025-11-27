import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("../competition_package/datasets/train.parquet")

feature_cols = [c for c in df.columns if c not in ['seq_ix', 'step_in_seq', 'need_prediction']]

# ============================
# 1. Sequence Integrity Check
# ============================
seq_lengths = df.groupby('seq_ix').size()
print(seq_lengths.describe())
print("Sequences not equal to 1000:", (seq_lengths != 1000).sum())

missing_steps = df.groupby('seq_ix')['step_in_seq'].apply(
    lambda x: set(range(1000)) - set(x)
)
print("Sequences with missing steps:", sum(len(v) > 0 for v in missing_steps))

# ============================
# 2. need_prediction pattern
# ============================
pred_steps = df[df["need_prediction"] == 1].step_in_seq.value_counts().sort_index()
print(pred_steps.head(20))

# ============================
# 3. Per-feature stats
# ============================
desc = df[feature_cols].describe().T
desc['missing_pct'] = df[feature_cols].isna().mean() * 100
print(desc)

# ============================
# 4. Per-feature variance
# ============================
variance = df[feature_cols].var().sort_values()
print("Lowest variance features:\n", variance.head(20))
print("Highest variance features:\n", variance.tail(20))

# ============================
# 5. Lag-1 correlation feature(t) vs feature(t+1)
# ============================
lag1_corr = {}

for col in feature_cols:
    x = df[col].values
    lag1_corr[col] = np.corrcoef(x[:-1], x[1:])[0,1]

lag1_corr = pd.Series(lag1_corr).sort_values()
print("Lowest lag-1 correlations:\n", lag1_corr.head(20))
print("Highest lag-1 correlations:\n", lag1_corr.tail(20))

# ============================
# 6. Autocorrelation (ACF) for first 10 features
# ============================
from statsmodels.tsa.stattools import acf

for col in feature_cols[:10]:
    ac = acf(df[col].values, nlags=20, fft=True)
    plt.plot(ac, label=col)

plt.title("Autocorrelation (first 10 features)")
plt.legend()
plt.savefig("visualisations/acf_first_10_features.png")
plt.show()

# ============================
# 7. Correlation Matrix (first 30 features)
# ============================
subset = feature_cols
corr = df[subset].corr()

plt.figure(figsize=(12,10))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Correlation matrix")
plt.savefig("visualisations/corrmatrix.png")
plt.show()

# ============================
# 8. Feature drift across steps
# ============================
mean_by_step = df.groupby('step_in_seq')[feature_cols].mean()

plt.figure(figsize=(10,5))
plt.plot(mean_by_step.iloc[:,0].values)
plt.title(f"Drift of Feature {feature_cols[0]} Across Steps")
plt.xlabel("step_in_seq")
plt.ylabel("mean")
plt.savefig("visualisations/drift.png")
plt.show()
