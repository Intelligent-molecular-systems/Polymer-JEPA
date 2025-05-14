import pandas as pd
df = pd.read_csv("dataset-augmented-original-combined-poly_chemprop.csv")
df_na_free = df.dropna(axis=0, subset=['EA vs SHE (eV)'])
only_na = df[~df.index.isin(df_na_free.index)]
only_na.to_csv("aldeghi_coley_augmented.csv")
