import pandas as pd

df = pd.read_parquet('data/sample_full.parquet')
print(df.shape)

df = pd.read_parquet('data/sample_train.parquet')
print(df.shape)

df = pd.read_parquet('data/sample_val.parquet')
print(df.shape)

df = pd.read_parquet('data/tiny_train.parquet')
print(df.shape)

df = pd.read_parquet('data/tiny_val.parquet')
print(df.shape)