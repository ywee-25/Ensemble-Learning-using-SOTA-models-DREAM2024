import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

DIR_DATA = Path(__file__).parent/ "data"

def averaging_ensemble(all_pred):
    avg_pred = np.mean(all_pred, axis=1)

    return avg_pred

if __name__ == "__main__":
    y_true = pd.read_csv(DIR_DATA / 'leaderboard_targets.csv')['Experimental_value']
    #name_list = ['belfaction','CASI_Leaderboard_set_Submission' 'D2Smell']
    name_list = ['belfaction','D2Smell']
    lb_preds = []
    for name in name_list:
        pred = pd.read_csv(DIR_DATA / f'{name}_Leaderboard_set_Submission.csv')['Predicted_Experimental_Values']
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        pearson = pearsonr(y_true, pred)[0]
        print(f"{name:<15} RMSE: {rmse:.5f}  Pearson: {pearson:.5f}")
        lb_preds.append(pred.values)
    lb_preds = np.array(lb_preds).T  # (n_samples, n_models)
    y_averaging = averaging_ensemble(lb_preds)


    rmse = np.sqrt(mean_squared_error(y_true, y_averaging))
    pearson = pearsonr(y_true, y_averaging)[0]
    print(f"Averaging Ensemble RMSE: {rmse:.5f}, Pearson: {pearson:.5f}")

    # belfaction
    # RMSE: 0.11281
    # Pearson: 0.72192
    # CASI
    # RMSE: 0.15300
    # Pearson: 0.33423   ?might be some problems with the submission data given
    # D2Smell
    # RMSE: 0.07426
    # Pearson: 0.92476
    # Averaging
    # Ensemble
    # RMSE: 0.10091, Pearson: 0.85272

    # belfaction
    # RMSE: 0.11281
    # Pearson: 0.72192
    # D2Smell
    # RMSE: 0.07426
    # Pearson: 0.92476
    # Averaging
    # Ensemble
    # RMSE: 0.08155, Pearson: 0.90167 # performance no better than D2Smell





