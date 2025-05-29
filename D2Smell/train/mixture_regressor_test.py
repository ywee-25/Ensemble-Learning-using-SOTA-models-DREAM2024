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
import optuna
from pathlib import Path
DIR_DATA = Path(__file__).parents[1] / "data"
DIR_OUTPUT = Path(__file__).parents[1] / "output"

seed = 3407  # https://arxiv.org/abs/2109.08203


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

    training_data_df = training_data_df.copy()
    training_data_df['Adjusted_Exp_Values'] = training_data_df.apply(
        adjust_exp_value, axis=1, args=(add_aug_val,))

    y = training_data_df['Adjusted_Exp_Values'].values

    return y


def prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df, all_cids):
    X_test = []

    for _, row in leaderboard_submission_df.iterrows():
        dataset, mixture1, mixture2 = row['Dataset'], row['Mixture_1'], row['Mixture_2']

        mixture1_vector = prepare_binary_vector(
            mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids,
            mixture1)
        mixture2_vector = prepare_binary_vector(
            mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids,
            mixture2)

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1),
                    'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2),
                    'Prediction_1':].to_numpy().flatten()

        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)

    return np.array(X_test)


def prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids):
    X_test = []

    for _, row in test_set_submission_df.iterrows():
        dataset, mixture1, mixture2 = 'Test', row['Mixture_1'], row['Mixture_2']

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1),
                    'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2),
                    'Prediction_1':].to_numpy().flatten()
        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        mixture1_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture2)

        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)

    return np.array(X_test)


def train_model(training_data_df, df_percept, mixture_definitions_df, all_cids, X_t, y_t, data_percept, percept_ids):
    best_t_mse = 100
    best_t_corr = -100
    best_models = []
    best_weights = []
    best_oof = None
    best_all_folds_preds = None
    all_df, X = prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids)


    X = expand_features(X, data_percept, percept_ids)
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X = poly.fit_transform(X)

    n_folds = 10

    def objective(trial):
        nonlocal best_t_mse
        nonlocal best_t_corr
        nonlocal best_models
        nonlocal best_weights
        nonlocal best_oof
        nonlocal best_all_folds_preds

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'min_child_weight': trial.suggest_float('min_child_weight', 6, 11),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.4),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'add_aug_val': trial.suggest_float('add_aug_val', 0.000001, 0.001),
            'remove_aug_val': trial.suggest_float('remove_aug_val', 0.001, 0.1)
        }

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            device = 'cuda',
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            reg_lambda=params['reg_lambda'],
            reg_alpha=params['reg_alpha'],
            verbosity=2,
            #predictor='gpu_predictor',
        )

        y = prepare_training_data_aug(all_df, params['add_aug_val'], params['remove_aug_val'])

        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        models = []
        scores = []
        oof = np.zeros_like(y)
        n_samples = X.shape[0]

        all_folds_preds = np.zeros((n_samples, n_folds))


        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[test_index]
            train_df = all_df.iloc[train_index]
            val_df = all_df.iloc[test_index]

            y_train = prepare_training_data_aug(train_df, params['add_aug_val'], params['remove_aug_val'])
            y_val = prepare_training_data_aug(val_df, params['add_aug_val'], params['remove_aug_val'])

            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            oof[test_index] = y_pred
            all_folds_preds[:, fold] = xgb_model.predict(X)

            models.append(xgb_model)
            scores.append(score)

        weights = [1 / score for score in scores]
        weights = [weight / sum(weights) for weight in weights]

        y_t_pred = predict_with_average_model(models, weights, X_t)
        t_corr = pearson_corr(y_t, y_t_pred)

        if t_corr > best_t_corr:
            best_t_corr = t_corr
            best_models = models
            best_weights = weights
            best_oof = oof.copy()  # oof under best model
            best_all_folds_preds = all_folds_preds.copy()


        return t_corr  # Maximize correlation

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=20)

    feat_names = [f'feat_{i}' for i in range(X.shape[1])]
    df_out = pd.concat([
        pd.DataFrame(X, columns=feat_names),
        all_df[['Dataset', 'Mixture 1', 'Mixture 2', 'Experimental Values']].reset_index(drop=True),
        pd.Series(best_oof, name="oof_pred")
    ], axis=1)

    df_out.to_csv(DIR_OUTPUT / f"D2Smell_feats_and_oof.csv", index=False)

    return best_models, best_weights, best_all_folds_preds


def predict_with_average_model(models, weights, X):
    weighted_preds = np.zeros(X.shape[0])

    for model, weight in zip(models, weights):
        preds = model.predict(X)
        weighted_preds += weight * preds

    return weighted_preds


if __name__ == "__main__":

    print("Load data ... ... ", end='')
    cid_df = pd.read_csv(DIR_DATA / 'processed/CID.csv', header=[0])
    data_percept = pd.read_csv(DIR_OUTPUT / 'percept_scaled.csv')
    training_data_df = pd.read_csv(DIR_DATA / 'processed/gt_with_dataset_V2_augmented_dataset.csv')
    mixture_definitions_df = pd.read_csv(DIR_DATA / 'processed/Mixure_Definitions_augmented_dataset.csv')
    leaderboard_submission_df = pd.read_csv(DIR_DATA / 'raw/forms/Leaderboard_set_Submission_form.csv')
    mixture_definitions_leaderboard_df = pd.read_csv(DIR_DATA / 'raw/mixtures/Mixure_Definitions_Training_set_VS2.csv')
    test_set_submission_df = pd.read_csv(DIR_DATA / 'raw/forms/Test_set_Submission_form.csv')
    mixture_definitions_test_df = pd.read_csv(DIR_DATA / 'raw/mixtures/Mixure_Definitions_test_set.csv')
    df_percept = pd.read_csv(DIR_DATA / 'processed/predictions_separated_mean_33_Augmentation_Dataset.csv')
    true_values_ld_df = pd.read_csv(DIR_DATA / 'raw/forms/LeaderboardData_mixturedist.csv')
    print("Done")

    print("Prepare data ... ... ", end='')
    df_percept.drop(
        columns=['Prediction_11', 'Prediction_18', 'Prediction_21', 'Prediction_22', 'Prediction_27', 'Prediction_28',
                 'Prediction_31', 'Prediction_32'], inplace=True)
    df_percept = applying_2QuantileTransformer(df_percept)
    percept_ids = list(range(1, data_percept.shape[1]))
    all_cids = sorted(cid_df['CID'].astype(int).tolist())

    print("[leadboard data] ", end='')
    X_leaderboard = prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df,
                                           all_cids)
    X_leaderboard = expand_features(X_leaderboard, data_percept, percept_ids)
    X_leaderboard = applying_PolynomialFeatures(X_leaderboard)
    y_true = true_values_ld_df['Experimental Values'].values

    feat_names = [f'feat_{i}' for i in range(X_leaderboard.shape[1])]
    df_feats = pd.DataFrame(
        X_leaderboard,
        columns=feat_names
    )
    df_feats['Dataset'] = leaderboard_submission_df['Dataset'].values
    df_feats['Mixture_1'] = leaderboard_submission_df['Mixture_1'].values
    df_feats['Mixture_2'] = leaderboard_submission_df['Mixture_2'].values

    leaderboard_feature_path = DIR_OUTPUT / f"Leaderboard_feats_{seed}.csv"
    df_feats.to_csv(leaderboard_feature_path, index=False)
    print(f"Leaderboard features saved to {leaderboard_feature_path}")

    print("[test data] ", end='')
    X_test = prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids)
    X_test = expand_features(X_test, data_percept, percept_ids)
    X_test = applying_PolynomialFeatures(X_test)
    print("Done")

    print("Training ... ... ", end='')
    model, weight, oof_preds_all_folds = train_model(training_data_df, df_percept, mixture_definitions_df, all_cids, X_leaderboard, y_true,
                                data_percept, percept_ids)
    out_model_path = DIR_OUTPUT / f"mixture_model_{seed}.pkl"
    jl.dump({'model': model, 'weight': weight}, out_model_path)
    df_oof_all_folds = pd.DataFrame(oof_preds_all_folds)
    df_oof_all_folds.to_csv(DIR_OUTPUT / "D2Smell_train_oof_all_folds.csv", index=False)

    print("Inference [leadboard] ... ... ", end='')
    y_leaderboard_pred = predict_with_average_model(model, weight, X_leaderboard)
    leaderboard_submission_df['Predicted_Experimental_Values'] = y_leaderboard_pred
    submission_output_path = DIR_OUTPUT / f"Leaderboard_set_Submission_form_{seed}.csv"
    leaderboard_submission_df.to_csv(submission_output_path, index=False)
    true_values_ld_df = pd.read_csv(DIR_DATA / 'raw/forms/LeaderboardData_mixturedist.csv')
    y_true = true_values_ld_df['Experimental Values'].values
    rmse_value = rmse(y_true, y_leaderboard_pred)
    pearson_corr_value = pearson_corr(y_true, y_leaderboard_pred)
    print(f'Leaderboard RMSE: {rmse_value}')
    print(f'Leaderboard Pearson Correlation: {pearson_corr_value}')
    print("Done")

    print("Inference [test] ... ... ", end='')
    y_test_pred = predict_with_average_model(model, weight, X_test)
    test_set_submission_df['Predicted_Experimental_Values'] = y_test_pred
    test_set_submission_output_path = DIR_OUTPUT / f"Test_set_Submission_form_{seed}.csv"
    test_set_submission_df.to_csv(test_set_submission_output_path, index=False)
    print("Done")

