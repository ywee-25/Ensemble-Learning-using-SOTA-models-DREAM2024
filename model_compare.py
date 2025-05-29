import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

DIR_DATA = Path(__file__).parent/ "data"

def model_similarity(oof):

    corr_matrix = oof.corr(method='pearson')
    print(corr_matrix)
    return corr_matrix

    # plt.figure(figsize=(7, 6))
    # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', square=True,
    #             cbar_kws={"shrink": 0.8}, annot_kws={"size": 11},linewidths=0.5,linecolor='#e0e0e0')
    # plt.title("Model Prediction Pearson Correlation")
    # sns.despine()
    # plt.show()

if __name__ == "__main__":
    X_train = pd.read_csv(DIR_DATA / 'train_oofs.csv')   #(n_samples, n_models)
    y_train = pd.read_csv(DIR_DATA / 'TrainingData_mixturedist.csv')['Experimental Values'].dropna() #(n_samples,)
    X_val = pd.read_csv(DIR_DATA / 'lb_pred.csv')  #(n_val, n_models)
    y_val = pd.read_csv(DIR_DATA / 'leaderboard_targets.csv')['Experimental_value']   #(n_val,)

    model_similarity(X_train)

#Similarity output
#               model1_exp1  model1_exp5  model1_exp21  model1_exp23    model2
# model1_exp1      1.000000     0.942089      0.891351      0.864076  0.653803
# model1_exp5      0.942089     1.000000      0.888386      0.860202  0.668395
# model1_exp21     0.891351     0.888386      1.000000      0.943400  0.708273
# model1_exp23     0.864076     0.860202      0.943400      1.000000  0.677613
# model2           0.653803     0.668395      0.708273      0.677613  1.000000
