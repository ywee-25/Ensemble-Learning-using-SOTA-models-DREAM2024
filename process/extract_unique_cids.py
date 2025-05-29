import os
import sys
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm


path_mixture = 'data/raw/mixtures'
path_processed = 'data/processed'


file_path = osp.join(path_mixture, 'Mixure_Definitions_Training_set_VS2.csv')
   
df = pd.read_csv(file_path)
# Extracting all CID columns and combining them into a single series
cid_columns = df.filter(like='CID')
all_cids = pd.concat([cid_columns[col].dropna() for col in cid_columns], ignore_index=True)
all_cids = all_cids[all_cids != 0]
# Extracting all unique CIDs that have appeared in the dataset
all_unique_cids = all_cids.unique()
all_unique_cids = all_unique_cids.astype(int)
all_unique_cids_sorted = sorted(all_unique_cids)

# Display the results
all_unique_cids_df = pd.DataFrame(all_unique_cids_sorted, columns=['CID'])

all_unique_cids_df.to_csv(osp.join(path_processed,'CID.csv'), index=False)

