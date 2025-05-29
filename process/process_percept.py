import pandas as pd
import os.path as osp
import numpy as np

# Paths
path_percept = 'data/raw/percept'
path_processed = 'data/processed'

# Load data
percept_single_df = pd.read_csv(osp.join(path_percept, 'percept_single.csv'))
cid_df = pd.read_csv(osp.join(path_processed, 'CID.csv'))

# Ensure CID is numeric
cid_df['CID'] = cid_df['CID'].apply(pd.to_numeric, errors='coerce')
percept_single_df['CID'] = percept_single_df['CID'].apply(pd.to_numeric, errors='coerce')

# Filter data
filtered_percept_df = percept_single_df[percept_single_df['CID'].isin(cid_df['CID'].dropna())]

lb2_df = pd.read_csv(osp.join(path_percept, 'LBs2.txt'), sep='\t')

# Descriptor mapping
descriptor_mapping = {
    'INTENSITY/STRENGTH': 'INTENSITY',
    'VALENCE/PLEASANTNESS': 'PLEASANTNESS',
    'BAKERY': 'BAKERY',
    'SWEET': 'SWEET',
    'FRUIT': 'FRUIT',
    'FISH': 'FISH',
    'GARLIC': 'GARLIC',
    'SPICES': 'SPICES',
    'COLD': 'COLD',
    'SOUR': 'SOUR',
    'BURNT': 'BURNT',
    'ACID': 'ACID',
    'WARM': 'WARM',
    'MUSKY': 'MUSKY',
    'SWEATY': 'SWEATY',
    'AMMONIA': 'AMMONIA',
    'DECAYED': 'DECAYED',
    'WOOD': 'WOOD',
    'GRASS': 'GRASS',
    'FLOWER': 'FLOWER',
    'CHEMICAL': 'CHEMICAL'
}

# Apply mapping
lb2_df['descriptor'] = lb2_df['descriptor'].map(descriptor_mapping)

# Pivot to wide format
lb2_wide_df = lb2_df.pivot_table(index='#oID', columns='descriptor', values='value', aggfunc='first').reset_index()

# Rename index column
lb2_wide_df.rename(columns={'#oID': 'CID'}, inplace=True)

# Merge with an empty percept_single structured dataframe
lb2_formatted_df = pd.concat([lb2_wide_df, pd.DataFrame(columns=percept_single_df.columns)]).fillna(np.nan)

# Ensure column order
lb2_formatted_df = lb2_formatted_df[percept_single_df.columns]

# Ensure CID columns are numeric
cid_df['CID'] = pd.to_numeric(cid_df['CID'], errors='coerce')
lb2_formatted_df['CID'] = pd.to_numeric(lb2_formatted_df['CID'], errors='coerce')

# Filter formatted dataframe
filtered_lb2_df = lb2_formatted_df[lb2_formatted_df['CID'].isin(cid_df['CID'].dropna())]

# Merge percept_1 and percept_2 dataframes
combined_percept_df = pd.concat([filtered_percept_df, filtered_lb2_df], ignore_index=True)

# Ensure column order
combined_percept_df = combined_percept_df[percept_single_df.columns]

# Sort by CID
combined_percept_df = combined_percept_df.sort_values(by='CID').reset_index(drop=True)

# Save combined data
output_combined_percept_path = osp.join(path_processed, 'percept.csv')
combined_percept_df.to_csv(output_combined_percept_path, index=False)
