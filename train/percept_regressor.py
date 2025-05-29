import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib as jl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_scaler, fit_and_standardize_minmax, inv_standardize

    

def load_percept_data(path):
    df_descriptors = pd.read_csv(os.path.join(path, 'selected_descriptors.csv'))
    df_percept = pd.read_csv(os.path.join(path, 'percept.csv'))

    cids_with_labels = df_percept['CID'].values
    cids_all = df_descriptors['CID'].values

    mask_with_labels = np.isin(cids_all, cids_with_labels)
    mask_without_labels = ~mask_with_labels

    X = df_descriptors[mask_with_labels].drop(columns='CID').values
    Y = df_percept.drop(columns='CID').values
    Y = np.nan_to_num(Y, nan=0)

    X_unlabeled = df_descriptors[mask_without_labels].drop(columns='CID').values

    return X, Y, X_unlabeled, cids_all, mask_with_labels, mask_without_labels

if __name__ == "__main__":
    # Paths
    path_processed = 'data/processed'
    path_output = 'output'

    scaler_model_path = os.path.join(path_output, 'features_scaler.sav')
    features_scalar = load_scaler(scaler_model_path)

    # Load data
    features, percept, features_unlabeled, cids_all, mask_with_labels, mask_without_labels = load_percept_data(path_processed)

    # Standardize and reduce dimensions
    features_scaled = features_scalar.transform(features)
    features_unlabeled_scaled = features_scalar.transform(features_unlabeled)

    percept_scaled, percept_scaler = fit_and_standardize_minmax(percept)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', verbosity=2, gpu_id=0)
    model = MultiOutputRegressor(xgb_model)
    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    model.fit(features_scaled, percept_scaled)

    # Save MultiOutputRegressor model
    jl.dump(model, os.path.join(path_output, 'percept_model.pkl'))
    jl.dump(percept_scaler, os.path.join(path_output, 'percept_scaler.sav'))
    jl.dump(features_scalar, os.path.join(path_output, 'features_scaler.sav'))


    # Predict for unlabeled data
    predictions_scaled = model.predict(features_unlabeled_scaled)
    predictions_unscaled = inv_standardize(predictions_scaled, percept_scaler)

    # Save labeled and unlabeled data
    column_names = ['CID', 'INTENSITY', 'PLEASANTNESS', 'BAKERY', 'SWEET', 'FRUIT', 'FISH',
                    'GARLIC', 'SPICES', 'COLD', 'SOUR', 'BURNT', 'ACID', 'WARM', 'MUSKY',
                    'SWEATY', 'AMMONIA', 'DECAYED', 'WOOD', 'GRASS', 'FLOWER', 'CHEMICAL']

    # Labeled data
    df_labeled = pd.DataFrame(percept, columns=column_names[1:])
    df_labeled.insert(0, 'CID', cids_all[mask_with_labels])
    df_labeled = df_labeled.round(9)

    # Unlabeled data
    df_unlabeled = pd.DataFrame(predictions_unscaled, columns=column_names[1:])
    df_unlabeled.insert(0, 'CID', cids_all[mask_without_labels])
    df_unlabeled = df_unlabeled.round(9)

    # Combine and sort by CID
    df_combined = pd.concat([df_labeled, df_unlabeled]).sort_values(by='CID').reset_index(drop=True)
    df_combined.to_csv(os.path.join(path_output, 'percept_unscaled.csv'), index=False)

    # Scale labeled and unlabeled data
    df_labeled_scaled = pd.DataFrame(percept_scaled, columns=column_names[1:])
    df_labeled_scaled.insert(0, 'CID', cids_all[mask_with_labels])

    df_unlabeled_scaled = pd.DataFrame(predictions_scaled, columns=column_names[1:])
    df_unlabeled_scaled.insert(0, 'CID', cids_all[mask_without_labels])

    # Combine and sort by CID
    df_combined_scaled = pd.concat([df_labeled_scaled, df_unlabeled_scaled]).sort_values(by='CID').reset_index(drop=True)
    df_combined_scaled.to_csv(os.path.join(path_output, 'percept_scaled.csv'), index=False)

    # Additional check on all columns
    for col in column_names[1:]:
        print(f"{col} (labeled):", df_labeled[col].describe())
        print(f"{col} (unlabeled):", df_unlabeled[col].describe())

    print('FINISHED: Percept Regressor')