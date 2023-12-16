import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import polars as pl
from glob import glob
from tqdm import tqdm

author = "tatematsu"
exp_id = "exp112_finetune"

def mae_with_nan(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    not_nan = ~np.isnan(y_true) & ~np.isnan(y_pred)
    score = mean_absolute_error(np.array(y_true)[not_nan].clip(0, 1), np.array(y_pred)[not_nan].clip(0, 1))
    return score, not_nan.sum()

def get_score(y_true1,y_true2,y_pred1,y_pred2):
    score1,nn_cnt1 = mae_with_nan(y_true1,y_pred1)
    score2,nn_cnt2 = mae_with_nan(y_true2,y_pred2)
    score = np.average([score1,score2], weights=[nn_cnt1,nn_cnt2])
    return score

def main():
    ### val
    df_preds = []
    for f in range(5):
        df_pred = pl.read_csv(f"/kaggle/logs/{exp_id}/conv_gnn_l12/fold{f}/val_submission.csv")
        df_preds.append(df_pred)
    df_pred = pl.concat(df_preds)
    df_pred = df_pred.sort("id")
    df_pred = df_pred.to_pandas()
    gt = pd.read_parquet("/kaggle/input/val_gt.parquet")
    
    idx_valid = gt["SN_filter"]==1
    val_score = get_score(
        gt[idx_valid]["reactivity_2A3_MaP"],
        gt[idx_valid]["reactivity_DMS_MaP"],
        df_pred[idx_valid]["reactivity_2A3_MaP"],
        df_pred[idx_valid]["reactivity_DMS_MaP"],
    )
    
    print(f"val_score {val_score}")
    df_pred.to_parquet(f"/kaggle/logs/{exp_id}/conv_gnn_l12/{author + '_' + exp_id}_oof.parquet")
    
    ### submit
    paths = sorted(glob(f"/kaggle/logs/{exp_id}/conv_gnn_l12/*/submission.csv"))
    sum_df = None
    for file in tqdm(paths):
        df = pl.read_csv(file)
        if sum_df is None:
            sum_df = df
        else:
            sum_df += df
    average_df = sum_df / len(paths)
    average_df = average_df.to_pandas()
    average_df.to_parquet(f"/kaggle/logs/{exp_id}/conv_gnn_l12/{author + '_' + exp_id}_submission.parquet")

if __name__ == "__main__":
    main()





