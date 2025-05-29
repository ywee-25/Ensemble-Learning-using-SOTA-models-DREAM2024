import os
import sys
import numpy as np
import pandas as pd
import joblib as jl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import train.shap_calculation as shap_calculation
import shap
import argparse
import matplotlib.pyplot as plt

import xgboost as xgb

seed = None

def prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids):
    df_percept_reduced = df_percept.set_index(['Dataset', 'Mixture Label'])

    training_data_df = training_data_df.merge(
        df_percept_reduced.add_suffix('_1'), 
        left_on=['Dataset', 'Mixture 1'], 
        right_index=True
    ).merge(
        df_percept_reduced.add_suffix('_2'), 
        left_on=['Dataset', 'Mixture 2'], 
        right_index=True
    )

    feature_columns_1 = df_percept.columns[2:] + '_1'
    feature_columns_2 = df_percept.columns[2:] + '_2'
    z = (training_data_df[feature_columns_1].values - training_data_df[feature_columns_2].values) ** 2

    def get_binary_vector_mapping(mixture_definitions_df, all_cids):
        vector_map = {}
        for _, row in mixture_definitions_df.iterrows():
            mixture_id = row['Mixture Label']
            components = row.iloc[2:].dropna().astype(int)
            binary_vector = np.isin(all_cids, components)
            vector_map[(row['Dataset'], mixture_id)] = binary_vector.astype(float)
        return vector_map

    vector_mapping = get_binary_vector_mapping(mixture_definitions_df, all_cids)

    training_data_df['mixture1_vector'] = training_data_df.apply(
        lambda row: vector_mapping[(row['Dataset'], row['Mixture 1'])], axis=1)
    training_data_df['mixture2_vector'] = training_data_df.apply(
        lambda row: vector_mapping[(row['Dataset'], row['Mixture 2'])], axis=1)
    
    X = np.hstack([
        np.hstack(training_data_df['mixture1_vector'].values).reshape(len(training_data_df), -1),
        np.hstack(training_data_df['mixture2_vector'].values).reshape(len(training_data_df), -1),
        z
    ])

    return training_data_df, X


def prepare_training_data_aug(training_data_df, add_aug_val, remove_aug_val):
    def adjust_exp_value(row, add_aug_val):
        if row['augmentation_Action'] == 'add_1-2' or row['augmentation_Action'] == 'add_2-1':
            return row['Experimental Values'] - add_aug_val
        elif row['augmentation_Action'] == 'remove_1-2':
            return row['Experimental Values'] + remove_aug_val
        else:
            return row['Experimental Values']

    training_data_df['Adjusted_Exp_Values'] = training_data_df.apply(
        adjust_exp_value, axis=1, add_aug_val=add_aug_val)
    
    y = training_data_df['Adjusted_Exp_Values'].values

    return y


def prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df, all_cids):
    X_test = []

    for _, row in leaderboard_submission_df.iterrows():
        dataset, mixture1, mixture2 = row['Dataset'], row['Mixture_1'], row['Mixture_2']
        
        mixture1_vector = prepare_binary_vector(mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids, mixture2)

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1), 'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2), 'Prediction_1':].to_numpy().flatten()
        
        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)
    
    return np.array(X_test)


def prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids):
    X_test = []

    for _, row in test_set_submission_df.iterrows():
        dataset, mixture1, mixture2 = 'Test', row['Mixture_1'], row['Mixture_2']

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1), 'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2), 'Prediction_1':].to_numpy().flatten()
        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        mixture1_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture2)
        
        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)
    
    return np.array(X_test)

 
def prepare_model_to_train(training_data_df, df_percept, mixture_definitions_df, all_cids, data_percept, percept_ids, add_aug_val, remove_aug_val):
    all_df, X = prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids)
    X = expand_features(X, data_percept, percept_ids)
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X = poly.fit_transform(X)
    X = pd.DataFrame(X, columns=poly.get_feature_names_out())

    y = prepare_training_data_aug(all_df, add_aug_val, remove_aug_val)

    return X, y

def predict_with_average_model(models, weights, X):
    weighted_preds = np.zeros(X.shape[0])
    
    for model, weight in zip(models, weights):
        preds = model.predict(X)
        weighted_preds += weight * preds
    
    return weighted_preds

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_model(path):
    try:
        # Load the model with CPU settings
        model_dict = jl.load(path)
        
        # If loading successful, modify model parameters
        if isinstance(model_dict, dict) and 'model' in model_dict:
            model = model_dict['model']
            if isinstance(model, xgb.XGBModel):
                model.get_booster().set_param('predictor', 'cpu_predictor')
                model.get_booster().set_param('tree_method', 'hist')
        
        return model_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mixture Regressor")
    parser.add_argument("--seed", type=int, required=False, help="Set the global seed", default=42)
    args = parser.parse_args()

    seed = args.seed
    
    print("Load data ... ... ", end='')
    path_output = 'output'
    cid_df = pd.read_csv('data/processed/CID.csv', header=[0])
    data_percept = pd.read_csv('output/percept_scaled.csv')
    training_data_df = pd.read_csv('data/processed/gt_with_dataset_V2_augmented_dataset.csv')
    mixture_definitions_df = pd.read_csv('data/processed/Mixure_Definitions_augmented_dataset.csv')
    leaderboard_submission_df = pd.read_csv('data/raw/forms/Leaderboard_set_Submission_form.csv')
    mixture_definitions_leaderboard_df = pd.read_csv('data/raw/mixtures/Mixure_Definitions_Training_set_VS2.csv')
    test_set_submission_df = pd.read_csv('data/raw/forms/Test_set_Submission_form.csv')
    mixture_definitions_test_df = pd.read_csv('data/raw/mixtures/Mixure_Definitions_test_set.csv')
    df_percept = pd.read_csv('data/processed/predictions_separated_mean_33_Augmentation_Dataset.csv')
    true_values_ld_df = pd.read_csv('data/raw/forms/LeaderboardData_mixturedist.csv')
    print("Done")

    print("Prepare data ... ... ", end='')
    df_percept.drop(columns=['Prediction_11','Prediction_18','Prediction_21','Prediction_22','Prediction_27','Prediction_28','Prediction_31','Prediction_32'], inplace=True)
    df_percept = applying_2QuantileTransformer(df_percept)
    percept_ids = list(range(1, data_percept.shape[1]))
    all_cids = sorted(cid_df['CID'].astype(int).tolist())

    print("[leadboard data] ", end='')
    X_leaderboard = prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df, all_cids)
    X_leaderboard = expand_features(X_leaderboard, data_percept, percept_ids)
    X_leaderboard = applying_PolynomialFeatures(X_leaderboard)
    y_true = true_values_ld_df['Experimental Values'].values

    print("[test data] ", end='')
    X_test = prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids)
    X_test = expand_features(X_test, data_percept, percept_ids)  
    X_test = applying_PolynomialFeatures(X_test)
    print("Done")

    add_aug_val = 0
    remove_aug_val = 0
    print("Prepare model to train ... ... ", end='')
    X, y  = prepare_model_to_train(training_data_df, df_percept, mixture_definitions_df, all_cids, data_percept, percept_ids, add_aug_val, remove_aug_val)
    X_orig = X[pd.isna(training_data_df.augmentation_Action)]
    y_orig = y[pd.isna(training_data_df.augmentation_Action)]
    print("Done")

    Xy_dict = {'X_orig': X_orig, 'y_orig': y_orig}
    jl.dump(Xy_dict, os.path.join(path_output, f'Xy_dict.pkl'))
    seeds = [42, 1111, 2222, 3333, 3407, 5067, 6666, 7777, 8888, 9999]
    for seed in seeds:
        #model_dict = jl.load(os.path.join(path_output, f'mixture_model_{seed}.pkl'))
        model_dict = load_model(os.path.join(path_output, f'mixture_model_{seed}.pkl'))
        model, weight = model_dict['model'], model_dict['weight']
    
        explainers = []
        shap_values = []
        shap_value_w = np.zeros(X_orig.shape)
        for idx, m in enumerate(model):
            explainer = shap.TreeExplainer(m)
            explainers.append(explainer)
            shap_values.append(explainer.shap_values(X_orig))
            shap_value_w += weight[idx] * shap_values[idx]

        # Save the SHAP values
        shap_dict = {
            'shap_values': shap_values,
            'shap_value_w': shap_value_w,
            'feature_names': X_orig.columns
        }
        jl.dump(shap_dict, os.path.join(path_output, f'shap_values_{seed}.pkl'))

        # Create plots with proper feature names
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_value_w, X_orig, feature_names=X_orig.columns, show=False)
        plt.savefig(os.path.join(path_output, f'shap_summary_weighted_{seed}.png'))
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[0], X_orig, feature_names=X_orig.columns, plot_type="bar", show=False)
        plt.savefig(os.path.join(path_output, f'shap_summary_bar_{seed}.png'))
        plt.close()

    
    print("SHAP calculation complete.")