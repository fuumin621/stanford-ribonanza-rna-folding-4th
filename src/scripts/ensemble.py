import numpy as np
import pandas as pd
import polars as pl
from glob import glob
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
pd.set_option('display.float_format', '{:.4f}'.format)

def load_data(df,paths):
    for path in paths:
        exp_id = path.split("/")[-1].split("_oof")[0]
        df_pred = pl.read_parquet(path).to_pandas()
        df[exp_id + "_" + "reactivity_2A3_MaP"] = df_pred["reactivity_2A3_MaP"].clip(0,1)
        df[exp_id + "_" + "reactivity_DMS_MaP"] = df_pred["reactivity_DMS_MaP"].clip(0,1)
    return df

def calculate_weights(X, y):
    def objective(weights):
        pred = np.dot(X, weights)
        return mean_absolute_error(y, pred)

    initial_weights = np.ones(X.shape[1])
    bounds = [(0, 1) for _ in range(X.shape[1])]
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds)
    return result.x

def process_predictions(df, target_cols):
    scores, nn_counts, all_weights = [], [], []
    for target_col in target_cols:
        pred_cols = [col for col in df.columns if (target_col in col) & (target_col!=col)]
        is_not_null = df[target_col].notnull()
        X = df.loc[is_not_null, pred_cols].values
        y = df.loc[is_not_null, target_col].values.clip(0, 1)
        
        weights = calculate_weights(X, y)
        pred = np.dot(X, weights)
        
        df.loc[is_not_null, f"{target_col}_pred"] = pred
        scores.append(mean_absolute_error(y, pred))
        nn_counts.append(is_not_null.sum())
        all_weights.append(weights)

    overall_score = np.average(scores, weights=nn_counts)
    return df, overall_score, all_weights

def prepare_submission(df, sub_paths, target_cols, all_weights):
    for path in sub_paths:
        exp_id = path.split("/")[-1].split("_submission")[0]
        df_pred = pl.read_parquet(path).to_pandas()
        df[exp_id + "_" + "reactivity_2A3_MaP"] = df_pred["reactivity_2A3_MaP"].clip(0,1)
        df[exp_id + "_" + "reactivity_DMS_MaP"] = df_pred["reactivity_DMS_MaP"].clip(0,1)
    
    for idx, target_col in enumerate(target_cols):
        pred_cols = [col for col in df.columns if (target_col in col) & (target_col!=col)]
        df[target_col] = np.dot(df[pred_cols].values, all_weights[idx])
    
    return df

def main():
    pred_dir = "/kaggle/input/predictions/"
    output_path = f"/kaggle/logs/ensemble/weighted_avg_final2_all_submission.parquet"
    # print(sorted(glob(pred_dir + "*oof.parquet")))

    oof_paths = [
        # with out PL
        pred_dir + 'tatematsu_exp112_finetune_oof.parquet',
        pred_dir + 'tatematsu_exp300_finetune_oof.parquet',
        pred_dir + 'tatematsu_exp302_finetune_oof.parquet',
        pred_dir + 'tattaka_exp064_oof.parquet',
        pred_dir + 'tattaka_exp071_oof.parquet',
        pred_dir + 'yu4u_yu4u_oof.parquet',

        # with PL
        pred_dir + 'tatematsu_exp312_finetune_oof.parquet',
        pred_dir + 'tatematsu_exp317_finetune_oof.parquet',
        pred_dir + 'tattaka_exp070_pl_oof.parquet',
        pred_dir + 'tattaka_exp070_tiny_pl_oof.parquet',
        pred_dir + 'tattaka_exp072_pl_oof.parquet',
        pred_dir + 'yu4u_yu4upl2_oof.parquet',
    ]

    target_cols = ['reactivity_DMS_MaP', 'reactivity_2A3_MaP']

    df = pd.read_parquet("/kaggle/input/val_gt.parquet")
    df = load_data(df,oof_paths)
    df[target_cols] = df[target_cols].clip(0,1)
    df = df[df["SN_filter"]==1].reset_index(drop=True)

    # val score check
    cv_scores = []
    exp_ids = [path.split("/")[-1].split("_oof")[0] for path in oof_paths]
    for exp_id in exp_ids:
        pred_cols = [exp_id + "_" + "reactivity_DMS_MaP",exp_id + "_" + "reactivity_2A3_MaP"]
        score = np.nanmean(np.abs(df[target_cols].values.clip(0, 1) - df[pred_cols].values.clip(0, 1)))
        cv_scores.append(score)

    df, overall_score, all_weights = process_predictions(df, target_cols)

    exp_ids = [p.split("/")[-1].split("_oof")[0] for p in oof_paths]
    results = pd.DataFrame(exp_ids,columns=["exp_id"])
    results["cv_score"] = cv_scores
    results["weights_DMS"] = all_weights[0]
    results["weights_2A3"] = all_weights[1]
    print("Overall Score:", overall_score)
    print(results)

    sub_paths = [p.replace("oof","submission") for p in oof_paths]
    submission_df = pl.read_csv('/kaggle/input/stanford-ribonanza-rna-folding/sample_submission.csv').to_pandas()
    submission_df = prepare_submission(submission_df, sub_paths, target_cols, all_weights)
    submission_df = submission_df[["id",'reactivity_DMS_MaP', 'reactivity_2A3_MaP']]
    submission_df.to_parquet(output_path)

    print(submission_df.head())

if __name__ == "__main__":
    main()

