import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import os.path as osp

import joblib as jl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.utils import fit_and_standardize


def load_descriptors(path, name):
    df_descriptors = pd.read_csv(osp.join(path, name))
    features = df_descriptors.drop(columns='CID').values
    features_scaled, features_scaler = fit_and_standardize(features)
    return features_scaled, df_descriptors.drop(columns='CID').columns, features_scaler


if __name__ == "__main__":
    # load data
    path_processed = 'data/processed' 
    features_scaled, feature_names, features_scaler = load_descriptors(path_processed, 'selected_descriptors.csv')
    
    path_output = 'output' 
    os.makedirs(path_output, exist_ok=True)
    jl.dump(features_scaler, os.path.join(path_output, f'features_scaler.sav'))
