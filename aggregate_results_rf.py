import pandas as pd

# Load regression csv
df = pd.read_csv('RF_results_test.csv')

# Group by 'finetune_percentage' and aggregate
agg_df = df.groupby('finetune_percentage').agg(
    r2_mean=('r2', 'mean'),
    r2_std=('r2', 'std'),
    rmse_mean=('rmse', 'mean'),
    rmse_std=('rmse', 'std')
).reset_index()

# Add 'real_percentage' column
agg_df['real_percentage in %'] = agg_df['finetune_percentage'] * 40

agg_df.to_csv('summary_RF_aldeghi.csv')

# Load classification csv
df_c = pd.read_csv('RF_results_diblock_test_stoich.csv')

# Group by 'finetune_percentage' and aggregate
agg_df_c = df_c.groupby('finetune_percentage').agg(
    prc_mean=('prc', 'mean'),
    prc_std=('prc', 'std'),
).reset_index()


agg_df_c.to_csv('summary_RF_diblock.csv')