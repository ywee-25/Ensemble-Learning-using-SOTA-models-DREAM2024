import numpy as np
import pandas as pd
from pathlib import Path
from heapq import merge

DIR_DATA = Path(__file__).parent / "data"
output_path = 'DIR_DATA'

def make_model1_oofs(exp_list):
    # process oofs of belfaction
    # keep the group labels(e.g. 'Dataset', 'Mixture 1'...) to match results from model2, which used data augmentation
    df = pd.read_csv(DIR_DATA/'TrainingData_mixturedist.csv').dropna(how = 'all')
    for exp in exp_list:
        oof = pd.read_csv(DIR_DATA / f"belfaction_oof_pred_exp{exp}.csv")['oof_pred']
        df[f'model1_exp{exp}'] = oof
    df['Mixture 1'] = df['Mixture 1'].astype(int)
    df['Mixture 2'] = df['Mixture 2'].astype(int)
    return df

def make_model1_exp23_feats():
    df = pd.read_csv(DIR_DATA/'belfaction_feats_and_oof_exp23.csv')
    df.rename(columns={'exp23_oof': 'model1_exp23'}, inplace=True)
    return df

def make_model2_oofs(path):
    # process oofs of D2Smell
    # Ravia is represented as Ravia 4 in model2
    # model2 uses its own mixture_dist (processed)
    df = pd.read_csv(DIR_DATA / path)
    df['Dataset'] = df['Dataset'].str.replace(r'Ravia 4', 'Ravia', regex=True)
    df.rename(columns={'oof_pred': 'model2'}, inplace=True)
    df['Mixture 1'] = df['Mixture 1'].astype(int)
    df['Mixture 2'] = df['Mixture 2'].astype(int)
    return df

def merge_df(model1_df,model2_df,keys):
    return model1_df.merge(model2_df, on = keys, how = 'left')

def make_leaderboard_preds(exp_list):
    df = pd.read_csv(DIR_DATA / 'Leaderboard_set_Submission_form.csv').dropna(how='all')
    for exp in exp_list:
        pred = pd.read_csv(DIR_DATA / f"belfaction_Leaderboard_set_Submission_form_exp{exp}.csv")
        df[f'model1_exp{exp}'] = pred['Predicted_Experimental_Values']

    df2 = pd.read_csv(DIR_DATA / 'D2Smell_Leaderboard_set_Submission.csv')
    df['model2'] = df2['Predicted_Experimental_Values']

    return df.iloc[:, -5:]

def make_leaderboard_feats_and_preds():
    df1_feats = pd.read_csv(DIR_DATA / 'belfaction_Leaderboard_feats_exp23.csv')  # without group labels
    df2_feats = pd.read_csv(DIR_DATA / 'D2Smell_Leaderboard_feats_3407.csv')
    df1_pred = pd.read_csv(DIR_DATA / f"belfaction_Leaderboard_set_Submission_form_exp23.csv")
    df2_pred = pd.read_csv(DIR_DATA / 'D2Smell_Leaderboard_set_Submission.csv')

    df1_feats['model1_exp23'] = df1_pred['Predicted_Experimental_Values']
    df = pd.concat([df1_feats, df2_feats], axis=1)
    df['model2'] = df2_pred['Predicted_Experimental_Values']
    return df

def generate_model_interaction_feat(model1_oof, model2_oof):
    meta_features = pd.DataFrame({
        'diff': model1_oof - model2_oof,
        'abs_diff': np.abs(model1_oof - model2_oof),
        'mean': 0.5 * (model1_oof + model2_oof),
        'product':model1_oof * model2_oof,
        'ratio' :model1_oof / model2_oof
    })
    return meta_features

def generate_uncertainty_feat(oof_preds_all, prefix='model'):
    oof_preds_all = oof_preds_all.copy()  # 避免修改原数据
    oof_preds_all = oof_preds_all.replace(0, np.nan)
    arr = oof_preds_all.values  # 转成np数组

    return pd.DataFrame({
        f'{prefix}_uncertainty_std': np.nanstd(arr, axis=1),
        f'{prefix}_uncertainty_range': np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1),
        f'{prefix}_uncertainty_min': np.nanmin(arr, axis=1),
        f'{prefix}_uncertainty_max': np.nanmax(arr, axis=1),
    })

if __name__ == "__main__":
    exp_list = [1, 5, 21, 23]
    merge_list = ['Dataset', 'Mixture 1', 'Mixture 2']

    # concat oofs derived from belfaction(exp1,5,21,23) & D2Smell
    train_oofs = merge_df(make_model1_oofs(exp_list), make_model2_oofs('D2Smell_oof_pred_seed3407.csv'), merge_list).drop(columns = [
        'Experimental Values_x','Experimental Values_y','Dataset','Mixture 1','Mixture 2'])
    train_oofs.to_csv(DIR_DATA/'train_oofs.csv', index=False)
    # concat leaderboard preds
    lb_pred = make_leaderboard_preds(exp_list)
    lb_pred.to_csv(DIR_DATA / 'lb_pred.csv', index=False)

    # concat oofs & original features derived from belfaction(exp23) & D2Smell
    # structure from left to right: feats of model1, oof of model1, feats of model2, oof of model2...
    train_feats_and_oofs =merge_df(make_model1_exp23_feats(),make_model2_oofs('D2Smell_feats_and_oof.csv'), merge_list).drop(columns = [
        'Experimental Values_x','Experimental Values_y','Dataset','Mixture 1','Mixture 2'])
    train_feats_and_oofs.to_csv(DIR_DATA / 'train_feats_oofs.csv', index=False)
    # concat leaderboard features-preds with the same order
    lb_feats_and_pred = make_leaderboard_feats_and_preds()
    lb_feats_and_pred.to_csv(DIR_DATA / 'lb_feats_and_pred.csv', index=False)

    # separately keep oofs of model2 which are matched by model1 for residual calculation
    model2_oof_matched = train_oofs['model2']  # 500 rows
    model2_oof_matched.to_csv(DIR_DATA / 'D2Smell_oof_matched.csv', index=False)


# not used
# model1_exp23_oof = train_oofs['model1_exp23'].values
# model2_oof = train_oofs['model2'].values
# lb_model1_pred = model1_lb_pred['Predicted_Experimental_Values'].values
# lb_model2_pred = model2_lb_pred['Predicted_Experimental_Values'].values
# train_model_inter_feat = generate_model_interaction_feat(model1_exp23_oof, model2_oof)
# train_model_inter_feat.to_csv(DIR_DATA/'train_model_inter_feat.csv', index=False)
# lb_model_inter_feat = generate_model_interaction_feat(lb_model1_pred,lb_model2_pred)
# lb_model_inter_feat.to_csv(DIR_DATA/'lb_model_inter_feet.csv', index=False)
#
#
# model1_oof_all_folds = pd.read_csv(DIR_DATA/'belfaction_train_oof_all_folds_exp23.csv')
# model2_oof_all_folds = pd.read_csv(DIR_DATA/'D2Smell_train_oof_all_folds.csv')
#
# uncertainty_model1 = generate_uncertainty_feat(model1_oof_all_folds.iloc[:, :10], prefix='model1')
# uncertainty_model2 = generate_uncertainty_feat(model2_oof_all_folds.iloc[:, :10], prefix='model2')
# for col in merge_list:
#     uncertainty_model1[col] = model1_oof_all_folds[col] # add group labels for matching
#     uncertainty_model2[col] = model2_oof_all_folds[col]
#
# uncertainty_model1['Dataset'] = uncertainty_model1['Dataset'].str.replace(r'Ravia 4', 'Ravia', regex=True)
# uncertainty = uncertainty_model1.merge(uncertainty_model2, on = merge_list, how = 'left')
# uncertainty = uncertainty.drop(columns = ['Dataset','Mixture 1','Mixture 2'])
# uncertainty.to_csv(DIR_DATA/'train_uncertainty.csv', index=False)








