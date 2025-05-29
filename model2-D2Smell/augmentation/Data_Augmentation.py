import pandas as pd
import numpy as np
import itertools
import os

# Load existing data
training_data_df = pd.read_csv('gt_with_dataset_Original.csv')
mixture_definitions_df = pd.read_csv('Mixure_Definitions_Original.csv')

# Select columns that start with "CID_"
cid_columns = [col for col in mixture_definitions_df.columns if col.startswith('CID_')]

# Gather all unique CIDs from these columns, excluding zeros
unique_cids = pd.unique(mixture_definitions_df[cid_columns].values.ravel('K'))
unique_cids = unique_cids[unique_cids != 0]  # Exclude zero values
unique_cids=pd.Series(unique_cids)
unique_cids = unique_cids.dropna()
all_cids = sorted(unique_cids.astype(int).tolist())
print(f'list of unique CIDs: {all_cids}')

# prepare_binary_vector, data_aug_add, and data_aug_remove must be defined
def prepare_binary_vector(mixture_definitions_df, all_cids, mixture_id):
    binary_vector = np.zeros(len(all_cids))
    mixture_row = mixture_definitions_df[mixture_definitions_df['Mixture Label'] == mixture_id].iloc[:, 2:].values.flatten()
    mixture_components = mixture_row[~pd.isna(mixture_row)].astype(int)
    for cid in mixture_components:
        if cid in all_cids:
            binary_vector[all_cids.index(cid)] = 1
    return binary_vector

def data_aug_add(mix1, mix2, exp_val, aug_val, aug_num_list):
    new_mix1_list = []
    new_exp_val_list = []
    idx = np.where((mix2 == 1) & (mix1 == 0))[0]

    for aug_num in aug_num_list:
        idx_combinations = list(itertools.combinations(idx, aug_num))
        
        for combination in idx_combinations:
            new_mix1 = mix1.copy()
            new_exp_val = exp_val - aug_val * aug_num
            
            for i in combination:
                new_mix1[i] = 1
            
            new_mix1_list.append(new_mix1)
            new_exp_val_list.append(new_exp_val)
    
    return new_mix1_list, new_exp_val_list

def data_aug_remove(mix1, mix2, exp_val, aug_val, aug_num_list):
    new_mix1_list = []
    new_mix2_list = []
    new_exp_val_list = []

    idx = np.where((mix1 == 1) & (mix2 == 1))[0]

    for aug_num in aug_num_list:
        idx_combinations = list(itertools.combinations(idx, aug_num))
        
        for combination in idx_combinations:
            new_mix1 = mix1.copy()
            new_mix2 = mix2.copy()
            new_exp_val = exp_val + aug_val * aug_num
            
            for i in combination:
                new_mix1[i] = 0
                new_mix2[i] = 0
            
            new_mix1_list.append(new_mix1)
            new_mix2_list.append(new_mix2)
            new_exp_val_list.append(new_exp_val)
    
    return new_mix1_list, new_mix2_list, new_exp_val_list


def augment_and_update_definitions(training_data_df, mixture_definitions_df, all_cids):
    new_definitions = []
    augmented_rows = []
    new_mixture_id = max(mixture_definitions_df['Mixture Label']) + 1  # Assuming 'Mixture Label' is numeric

    for _, row in training_data_df.iterrows():
        dataset = row['Dataset']
        mixture1 = row['Mixture 1']
        mixture2 = row['Mixture 2']
        exp_value = row['Experimental Values']

        # Prepare binary vectors for both mixtures
        mixture1_vector = prepare_binary_vector(mixture_definitions_df[mixture_definitions_df['Dataset'] == dataset], all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_df[mixture_definitions_df['Dataset'] == dataset], all_cids, mixture2)

        # Generate augmented data by adding components to mixture1 from mixture2
        new_mix1_list, new_exp_val_list1 = data_aug_add(mixture1_vector, mixture2_vector, exp_value, 0.0, [1])
        
        # Generate augmented data by adding components to mixture2 from mixture1
        new_mix2_list, new_exp_val_list2 = data_aug_add(mixture2_vector, mixture1_vector, exp_value, 0.0, [1])
        
        # Generate augmented data by removing common components from both mixtures
        new_mix1_remove_list, new_mix2_remove_list, new_exp_val_remove_list = data_aug_remove(mixture1_vector, mixture2_vector, exp_value, 0.0, [1])
        combined_lists = zip(new_mix1_list + new_mix2_list + new_mix1_remove_list, 
                             new_exp_val_list1 + new_exp_val_list2 + new_exp_val_remove_list,
                             [1]*len(new_mix1_list) + [3]*len(new_mix2_list) + [5]*len(new_mix1_remove_list))

        for new_mix, new_val, action_code in combined_lists:
            # Determine the Augmentation action based on the action_code
            if action_code == 1 or action_code == 1:
                action = 'add_1-2'
            elif action_code == 3:
                action = 'add_2-1'
            elif action_code == 5:
                action = 'remove_1-2'
           
            # Determine 'Mixture 2' based on action_code
            if action_code == 1 or action_code == 5:
                mixture_2_value = mixture2
            elif action_code == 3:
                mixture_2_value = mixture1    

            augmented_row = {
                'Dataset': dataset,
                'Mixture 1': new_mixture_id,
                'Mixture 2': mixture_2_value,
                'Experimental Values': new_val,
                'augmentation_Action': action
            }
            augmented_rows.append(augmented_row)
            
            # Create a new definition entry
            new_def_row = {
                'Mixture Label': new_mixture_id,
                'Dataset': dataset
            }

            # Assign CIDs to new definition starting from the first CID column
            component_index = 1
            for i, presence in enumerate(new_mix):
                if presence == 1:
                    cid_key = f'CID_{component_index}'
                    new_def_row[cid_key] = all_cids[i]
                    component_index += 1

            new_definitions.append(new_def_row)
            new_mixture_id += 1  # Increment for the next new mixture

    new_definitions_df = pd.DataFrame(new_definitions)
    augmented_training_data = pd.DataFrame(augmented_rows)

    return augmented_training_data, new_definitions_df

# Perform augmentation and get new definitions
augmented_training_data, new_definitions_df = augment_and_update_definitions(training_data_df, mixture_definitions_df, all_cids)

# Append augmented data to the original DataFrame
new_training_data_df = pd.concat([training_data_df, augmented_training_data], ignore_index=True)
new_mixture_definitions_df = pd.concat([mixture_definitions_df, new_definitions_df], ignore_index=True)

# Save the augmented_dataset to CSV
new_training_data_df.to_csv('gt_with_dataset_V2_augmented_dataset.csv', index=False)
                            
new_mixture_definitions_df.to_csv('Mixure_Definitions_augmented_dataset.csv', index=False)

print("Augmented data and new mixture definitions saved successfully!")
