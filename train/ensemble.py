import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib as jl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from pathlib import Path
import argparse

DIR_DATA = Path(__file__).parents[1] / "data"
DIR_OUTPUT = Path(__file__).parents[1] / "output"


def load_models_and_weights(model_dir, model_prefix):
    models_and_weights = []
    for file_name in os.listdir(model_dir):
        if file_name.startswith(model_prefix) and file_name.endswith('.pkl'):
            model_path = os.path.join(model_dir, file_name)
            model_data = jl.load(model_path)
            models_and_weights.append(model_data)
    return models_and_weights

def predict_with_ensemble(models_and_weights, X):
    predictions = np.zeros((len(models_and_weights), len(X)))

    for i, model_data in enumerate(models_and_weights):
        print(i, end=' ')
        model_list = model_data['model']  
        weight_list = model_data['weight'] 
        
        sub_predictions = np.zeros(len(X))
        for model, weight in zip(model_list, weight_list):
            sub_predictions += model.predict(X) * weight
        
        predictions[i] = sub_predictions

    return np.mean(predictions, axis=0)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mixture Regressor")
    args = parser.parse_args()
    
    print("Load data ... ... ", end='')
    cid_df = pd.read_csv(DIR_DATA/'processed/CID.csv', header=[0])
    data_percept = pd.read_csv(DIR_OUTPUT/'percept_scaled.csv')
    training_data_df = pd.read_csv(DIR_DATA/'processed/gt_with_dataset_V2_augmented_dataset.csv')
    mixture_definitions_df = pd.read_csv(DIR_DATA/'processed/Mixure_Definitions_augmented_dataset.csv')
    leaderboard_submission_df = pd.read_csv(DIR_DATA/'raw/forms/Leaderboard_set_Submission_form.csv')
    mixture_definitions_leaderboard_df = pd.read_csv(DIR_DATA/'raw/mixtures/Mixure_Definitions_Training_set_VS2.csv')
    test_set_submission_df = pd.read_csv(DIR_DATA/'raw/forms/Test_set_Submission_form.csv')
    mixture_definitions_test_df = pd.read_csv(DIR_DATA/'raw/mixtures/Mixure_Definitions_test_set.csv')
    df_percept = pd.read_csv(DIR_DATA/'processed/predictions_separated_mean_33_Augmentation_Dataset.csv')
    true_values_ld_df = pd.read_csv(DIR_DATA/'raw/forms/LeaderboardData_mixturedist.csv')
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

    print("Load and Ensemble Models ... ... ", end='')
    model_prefix = 'mixture_model_'
    models_and_weights = load_models_and_weights(str(DIR_OUTPUT), model_prefix)
    print(f"Loaded {len(models_and_weights)} models for ensemble.")
    print("Done")

    print("Inference [leaderboard] with ensemble ... ... ", end='')
    y_leaderboard_pred = predict_with_ensemble(models_and_weights, X_leaderboard)
    leaderboard_submission_df['Predicted_Experimental_Values'] = y_leaderboard_pred
    submission_output_path = DIR_OUTPUT / f'Leaderboard_set_Submission_form_ensemble.csv'
    leaderboard_submission_df.to_csv(submission_output_path, index=False)
    rmse_value = rmse(y_true, y_leaderboard_pred)
    pearson_corr_value = pearson_corr(y_true, y_leaderboard_pred)
    print(f'Leaderboard RMSE (ensemble): {rmse_value}')
    print(f'Leaderboard Pearson Correlation (ensemble): {pearson_corr_value}')
    print("Done")

    print("Inference [test] with ensemble ... ... ", end='')
    y_test_pred = predict_with_ensemble(models_and_weights, X_test)
    test_set_submission_df['Predicted_Experimental_Values'] = y_test_pred
    test_set_submission_output_path = DIR_OUTPUT/ 'Test_set_Submission_form_ensemble.csv'
    test_set_submission_df.to_csv(test_set_submission_output_path, index=False)
    print("Done")

    