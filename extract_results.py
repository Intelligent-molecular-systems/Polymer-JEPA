import os
import pandas as pd
import re

# Folder where your csv files are stored
folder_path = './'  # Change this to your folder

# Pattern to extract values from the filename
filename_pattern = re.compile(
    r'metrics_test_PL_(?P<PL>.*?)_layer_norm_(?P<norm>\d+)_seeds_(?P<seeds>\d+)_finetune_percentage_(?P<percentage>[\d.]+)_pretraining_(?P<pretraining>.*?).csv'
)

# Collect all the data
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = filename_pattern.match(filename)
        if match:
            # Extract metadata from the filename
            metadata = match.groupdict()
            
            # Load the CSV content
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, header=None, names=['R2', 'RMSE'])

            # Add metadata to each row
            for _, row in df.iterrows():
                all_data.append({
                    'R2': row['R2'],
                    'RMSE': row['RMSE'],
                    **metadata
                })

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Remove rows where 'Metric' column contains things like 'R2' or 'RMSE'
unwanted_metrics = ['R2', 'RMSE']
final_df = final_df[~final_df['R2'].isin(unwanted_metrics)]

# Save to CSV
final_df.to_csv('combined_metrics.csv', sep=';', decimal='.', index=False)


# Load the CSV file using semicolon as the delimiter.
df = pd.read_csv('combined_metrics.csv', delimiter=';')

# Standardize column names
df.columns = df.columns.str.strip()

# Convert R2 and RMSE to numeric
df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')

# Standardize the grouping columns.
# For booleans, first convert to string, strip, then map to bool.
df['PL'] = df['PL'].astype(str).str.strip().map({'True': True, 'False': False})
df['pretraining'] = df['pretraining'].astype(str).str.strip().map({'True': True, 'False': False})

# For 'norm' and 'percentage', ensure they are numeric.
df['norm'] = pd.to_numeric(df['norm'], errors='coerce')
df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')

# Optionally, you can also standardize 'seeds' though we won't group by it:
df['seeds'] = pd.to_numeric(df['seeds'], errors='coerce')

# Check dtypes to be sure everything is as expected.
print(df.dtypes)
#print(df.head())

# Now, group by the columns that define the experimental configuration.
group_keys = ['PL', 'norm', 'percentage', 'pretraining']

summary = df.groupby(group_keys, as_index=False).agg(
    R2_mean=('R2', 'mean'),
    R2_std=('R2', 'std'),
    RMSE_mean=('RMSE', 'mean'),
    RMSE_std=('RMSE', 'std')
)

# Save the aggregated summary to a CSV using semicolon as the delimiter.
summary.to_csv('summary_statistics.csv', sep=';', index=False)