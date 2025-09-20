#!/usr/bin/env python
# Modified script from https://github.com/coleygroup/polymer-chemprop-data/blob/main/results/vipea/train_test_rf.py
# The pickle data files are generated according to https://github.com/coleygroup/polymer-chemprop-data/tree/main/datasets/vipea/make-polymer-fps.py

import collections
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.config import cfg, update_cfg

import numpy as np
import pandas as pd
import pickle


if __name__ == '__main__':
    cfg = update_cfg(cfg)

    
    with open(f'Data/dataset-poly_fps_counts.pkl', 'rb') as f:
        data = pickle.load(f)

    X_full = data['X']
    Y_full = data['Y']['EA vs SHE (eV)'].values  # Use only EA
    # --- Precompute fingerprints and targets ---
    # Define KFold split once
    # K-Fold splitting
    kf = KFold(n_splits=cfg.runs, shuffle=True, random_state=12345)
    splits = list(kf.split(X_full))

    # Define seeds and FT percentages, same as in JEPA setup for reproducibility
    aldeghiFTPercentages = [0.01, 0.02, 0.04, 0.1, 0.2]
    seed_sets = {
        0: [42, 123, 777, 888, 999],
        1: [421, 1231, 7771, 8881, 9991],
        2: [422, 1232, 7772, 8882, 9992]
    }
    metrics = collections.defaultdict(list)
    metrics_test = collections.defaultdict(list)
    for cfg_seeds, seeds in seed_sets.items():
        print("Used seeds:", seeds)

        for ft_percentage in aldeghiFTPercentages:
            print(f"Running for fine-tune percentage: {ft_percentage}")

            

            for run_idx, (train_index, test_index) in enumerate(splits):
                print("----------------------------------------")
                print(f'Run {run_idx}/{cfg.runs - 1}')

                # Split
                X_train_full = X_full.iloc[train_index]
                Y_train_full = Y_full[train_index]
                X_test_full = X_full.iloc[test_index]
                Y_test_full = Y_full[test_index]

                val_idx, test_idx = train_test_split(
                    np.arange(len(X_test_full)), test_size=0.5, random_state=12345
                )
                X_val = X_test_full.iloc[val_idx]
                Y_val = Y_test_full[val_idx]
                X_test = X_test_full.iloc[test_idx]
                Y_test = Y_test_full[test_idx]

                # Sampling training subset
                random.seed(seeds[run_idx])
                np.random.seed(seeds[run_idx])
                subset_size = int(math.ceil(ft_percentage * len(X_train_full) / 64) * 64)
                selected_indices = np.random.choice(len(X_train_full), size=subset_size, replace=False)

                X_train = X_train_full.iloc[selected_indices]
                Y_train = Y_train_full[selected_indices]

                # Train RF
                rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=12)
                rf.fit(X_train, Y_train)

                # Predict
                Y_pred_train = rf.predict(X_train)
                Y_pred_val = rf.predict(X_val)
                Y_pred_test = rf.predict(X_test)

                # Metrics (only EA)
                r2_train = r2_score(Y_train, Y_pred_train)
                r2_val = r2_score(Y_val, Y_pred_val)
                r2_test = r2_score(Y_test, Y_pred_test)

                rmse_train = math.sqrt(mean_squared_error(Y_train, Y_pred_train))
                rmse_val = math.sqrt(mean_squared_error(Y_val, Y_pred_val))
                rmse_test = math.sqrt(mean_squared_error(Y_test, Y_pred_test))

                metrics['r2_train'].append(r2_train)
                metrics['r2_val'].append(r2_val)
                metrics['fold'].append(run_idx)
                metrics["seed_set"].append(cfg_seeds)
                metrics["finetune_percentage"].append(ft_percentage)
                metrics_test['r2'].append(r2_test)
                metrics_test['fold'].append(run_idx)
                metrics_test["seed_set"].append(cfg_seeds)
                metrics_test["finetune_percentage"].append(ft_percentage)


                metrics['rmse_train'].append(rmse_train)
                metrics['rmse_val'].append(rmse_val)
                metrics_test['rmse'].append(rmse_test)

    df = pd.DataFrame(metrics)
    df.to_csv(f'Results/experiments_paper/RF_results_train.csv', index=False)
    df = pd.DataFrame(metrics_test)
    df.to_csv(f'Results/experiments_paper/RF_results_test.csv', index=False)
