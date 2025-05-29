import numpy as np
import pandas as pd
from pathlib import Path
from heapq import merge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib import cm
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

DIR_DATA = Path(__file__).parent/ "data"
N_SPLITS = 5
RANDOM_STATE = 42

def quadratic_stacking_cv(X, y, n_splits=N_SPLITS, alpha=None, random_state=RANDOM_STATE):
    # scalers = {
    #     'StandardScaler': StandardScaler(),
    #     'MinMaxScaler': MinMaxScaler(),
    #     'MaxAbsScaler': MaxAbsScaler(),
    #     'RobustScaler': RobustScaler(),
    # }
    if alpha is None:
        alphas = np.logspace(-3, 2, 20)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses, pears, best_alphas = [], [], []

    for fold,(train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # X_tr_scaled = scaler.fit_transform(X_tr)
        # X_val_scaled = scaler.transform(X_val)
        # X_tr_poly = poly.fit_transform(X_tr_scaled)
        # X_val_poly = poly.transform(X_val_scaled)
        X_tr_poly = poly.fit_transform(X_tr)
        X_val_poly = poly.transform(X_val)

        grid = GridSearchCV(Ridge(),{'alpha':alphas},scoring = 'neg_root_mean_squared_error',cv=3)
        grid.fit(X_tr_poly, y_tr)
        best_alpha = grid.best_params_['alpha']
        best_alphas.append(best_alpha)

        meta_model = grid.best_estimator_
        y_val_pred = meta_model.predict(X_val_poly)

        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        pearson = pearsonr(y_val, y_val_pred)[0]
        rmses.append(rmse)
        pears.append(pearson)
        print(f"[Fold{fold+1}] Best alpha:{best_alpha:.5} | RMSE: {rmse:.5f}, Pearson: {pearson:.5f}")
    print(f"\nQuadratic Stacking using Ridge KFold Avg best alpha:{np.mean(best_alphas):.5f} | Avg RMSE: {np.mean(rmses):.5f}, Avg Pearson: {np.mean(pears):.5f}")
    return np.mean(best_alphas)

def quadratic_stacking_leaderboard(X, y, X_lb, y_lb, alpha=None):
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # X_scaled = scaler.fit_transform(X)
    # X_lb_scaled = scaler.transform(X_lb)
    # X_poly = poly.fit_transform(X_scaled)
    # X_lb_poly = poly.transform(X_lb_scaled)
    X_poly = poly.fit_transform(X)
    X_lb_poly = poly.transform(X_lb)

    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)

    # get weights
    coefs = model.coef_

    # if isinstance(X_train, pd.DataFrame):
    #     feature_names = X_train.columns
    # else:
    #     feature_names = [f"model_{i}" for i in range(X_train.shape[1])]
    #
    # for name, coef in zip(feature_names, coefs):
    #     print(f"{name:<20}  Weight: {coef:.5f}")

    y_lb_pred = model.predict(X_lb_poly)
    rmse = np.sqrt(mean_squared_error(y_lb, y_lb_pred))
    pearson = pearsonr(y_lb, y_lb_pred)[0]
    print(f"[Leaderboard] RMSE: {rmse:.5f}, Pearson: {pearson:.5f}")
    return y_lb_pred


def residual_stacking(X, y, X_val, y_val, model2_pred, random_state=RANDOM_STATE, return_rmse=False): # y should be residual

    residual_preds = np.zeros(len(X_val))

    model_residual = LGBMRegressor(n_estimators=100, random_state=random_state,verbose=-1)
    model_residual.fit(X, y)
    residual_pred = model_residual.predict(X_val)
    final_pred = model2_pred + residual_pred
    # scores
    rmse = np.sqrt(mean_squared_error(y_val, final_pred))
    pearson = pearsonr(y_val, final_pred)[0]
    print(f"Leaderboard RMSE: {rmse:.4f}, Pearson: {pearson:.4f}")

    importances = model_residual.feature_importances_
    feature_names = X.columns

    # N = 100
    # indices = importances.argsort()[::-1][:N]

    # plt.figure(figsize=(12, 8))
    # plt.bar(range(len(indices)), importances[indices])
    # plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    # plt.title("Top 100 Feature Importances in Meta Model")
    # plt.tight_layout()
    # plt.show()
    if return_rmse:
        return rmse, pearson, (importances, feature_names)
    return importances, feature_names

def compare_single_models(X_val, y_val):
    print("\n----- Single Model Performance on Leaderboard -----")
    if isinstance(X_val, pd.DataFrame):
        for col in X_val.columns:
            pred = X_val[col].values
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            pearson = pearsonr(y_val, pred)[0]
            print(f"{col:<25} RMSE: {rmse:.5f}, Pearson: {pearson:.5f}")
    else:
        for i in range(X_val.shape[1]):
            pred = X_val[:, i]
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            pearson = pearsonr(y_val, pred)[0]
            print(f"Model_{i:<20} RMSE: {rmse:.5f}, Pearson: {pearson:.5f}")

if __name__ == "__main__":
    X_train_oof = pd.read_csv(DIR_DATA / 'train_oofs.csv')  #(n_samples, n_models)
    y_train = pd.read_csv(DIR_DATA / 'TrainingData_mixturedist.csv')['Experimental Values'].dropna() #(n_samples,)
    X_lb_pred = pd.read_csv(DIR_DATA / 'lb_pred.csv')  #(n_lb, n_models)
    y_lb = pd.read_csv(DIR_DATA / 'leaderboard_targets.csv')['Experimental_value']  #(n_lb,)

    print("----- Quadratic Stacking KFold CV -----")
    selected_alpha = quadratic_stacking_cv(X_train_oof.iloc[:,-2:], y_train, n_splits=N_SPLITS)
    #
    print("----- Quadratic Stacking on Leaderboard-----")
    # #quadratic_stacking_leaderboard(X_train_oof, y_train, X_lb_oof, y_lb, alpha=selected_alpha)
    lb_pred_ridge = quadratic_stacking_leaderboard(
        X_train_oof.iloc[:,-2:], y_train, X_lb_pred.iloc[:,-2:], y_lb, alpha=1) #outperform

    # load features
    X_train_feat_oof = pd.read_csv(DIR_DATA / 'train_feats_oofs.csv')   # (500,21754)
    X_lb_feat_oof = pd.read_csv(DIR_DATA / 'lb_feats_and_pred.csv')         # (46,21754)
    # keep features separately for PCA
    X_train_feat = X_train_feat_oof.drop(columns = ['model1_exp23','model2'])
    X_lb_feat = X_lb_feat_oof.drop(columns = ['model1_exp23','model2'])
    # other features
    train_model_inter_feat = pd.read_csv(DIR_DATA / 'train_model_inter_feat.csv')  # (500,3)
    lb_model_inter_feat = pd.read_csv(DIR_DATA / 'lb_model_inter_feet.csv')
    #uncertainty_feat = pd.read_csv(DIR_DATA / 'train_uncertainty.csv') # (500,8)

    model1_exp23_oof = pd.read_csv(DIR_DATA / 'belfaction_oof_pred_exp23.csv')['oof_pred']
    model1_exp23_oof.name = 'model1_exp23'
    model1_exp23_pred = pd.read_csv(DIR_DATA / 'belfaction_Leaderboard_set_Submission_form_exp23.csv')[
        'Predicted_Experimental_Values']
    model1_exp23_pred.name = 'model1_exp23'
    model2_oof = pd.read_csv(DIR_DATA / 'D2Smell_oof_matched.csv').squeeze()
    model2_oof = pd.Series(model2_oof, name='model2')
    model2_pred = pd.read_csv(DIR_DATA / 'D2Smell_Leaderboard_set_Submission.csv')['Predicted_Experimental_Values']
    model2_pred = pd.Series(model2_pred, name='model2')

    print("----- Residual Stacking -----")
    pearson_records = {}
    def objective(trial):
        N_pca = trial.suggest_categorical('N_pca', [20, 50, 60, 100])
        N_select = trial.suggest_categorical('N_select', [20, 30, 40, 50, 60, 70, 100])

        # PCA
        pca = PCA(n_components=N_pca, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_feat)
        X_lb_pca = pca.transform(X_lb_feat)
        X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train_feat.index, columns=[f'pca_{i}' for i in range(N_pca)])
        X_lb_pca_df = pd.DataFrame(X_lb_pca, index=X_lb_feat.index, columns=[f'pca_{i}' for i in range(N_pca)])

        # concat features
        X_train_all = pd.concat([X_train_pca_df, model1_exp23_oof, model2_oof], axis=1)
        X_lb_all = pd.concat([X_lb_pca_df, model1_exp23_pred, model2_pred], axis=1)

        # use residual as label
        y_train_residual = y_train - model2_oof

        # return feature importances for selection
        importances, feature_names = residual_stacking(
            X_train_all, y_train_residual, X_lb_all, y_lb, model2_pred,
            random_state=RANDOM_STATE
        )
        # select top N features
        top_indices = importances.argsort()[::-1][:N_select]
        top_features = [feature_names[i] for i in top_indices]
        X_train_top = X_train_all[top_features]
        X_lb_top = X_lb_all[top_features]

        # return scores for tuning
        rmse, pearson, _ = residual_stacking(X_train_top, y_train_residual, X_lb_top, y_lb, model2_pred,
                                     random_state=RANDOM_STATE, return_rmse=True)
        pearson_records[trial.number] = {'pearson': pearson, 'params': (N_pca, N_select)}
        # minimize rmse
        return rmse

    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    best_pearson = pearson_records[best_trial.number]['pearson']
    print("Best params:", study.best_params)
    print(f"Best RMSE:{study.best_value:.5f}")
    print(f"Best Pearson: {best_pearson:.5f}")
