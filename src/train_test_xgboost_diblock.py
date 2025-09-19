#!/usr/bin/env python
# Modified script from https://github.com/coleygroup/polymer-chemprop-data/blob/main/results/vipea/train_test_rf.py
# The pickle data files are generated according to https://github.com/coleygroup/polymer-chemprop-data/tree/main/datasets/vipea/make-polymer-fps.py

import collections
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from config import cfg, update_cfg

if __name__ == '__main__':
    cfg = update_cfg(cfg)

    data = pd.read_pickle('Data/dataset-fps_counts-stoich_diblock.pkl')

    X_full = data['X']
    Y_full = data['Y']

    # Define seeds and FT percentages, same as in JEPA setup for reproducibility
    diblockFTPercentages = [0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.8]
    seed_sets = {
        0: [42, 123, 777, 888, 999],
        1: [421, 1231, 7771, 8881, 9991],
        2: [422, 1232, 7772, 8882, 9992]
    }
    metrics = collections.defaultdict(list)

    for cfg_seeds, seeds in seed_sets.items():
        print("Used seeds:", seeds)

        for ft_percentage in diblockFTPercentages:
            print(f"Running for fine-tune percentage: {ft_percentage}")

            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=12345)
            phase1_labels = Y_full.idxmax(axis=1).tolist()  # for stratification
            splits = list(sss.split(X_full, phase1_labels))

            for run_idx, (train_index, test_index) in enumerate(splits):
                print("----------------------------------------")
                print(f'Run {run_idx}/{cfg.runs - 1}')

                # Split
                X_train = X_full.iloc[train_index]
                Y_train = Y_full.iloc[train_index]
                X_test = X_full.iloc[test_index]
                Y_test = Y_full.iloc[test_index]

                # Sampling training subset
                random.seed(seeds[run_idx])
                np.random.seed(seeds[run_idx])
                subset_size = int(min((ft_percentage * len(X_full)), len(X_train)))
                selected_indices = np.random.choice(len(X_train), size=subset_size, replace=False)

                X_train_subset = X_train.iloc[selected_indices]
                Y_train_subset = Y_train.iloc[selected_indices]
                assert len(set(X_train_subset.index) & set(X_test.index)) == 0, "Data leakage: train and test overlap!"

                # Train XGBoost instead of RF
                model = XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",  # probability output for multi-class
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=12,
                    tree_method="hist"  # efficient training
                )

                # train model
                model.fit(X_train_subset, Y_train_subset.values.argmax(axis=1))

                # make predictions (probabilities for each class)
                Y_pred = model.predict_proba(X_test)

                # Convert to DataFrame with correct shape (n_samples x n_classes)
                df_predictions = pd.DataFrame(Y_pred, columns=Y_test.columns)

                # Metrics
                roc = roc_auc_score(Y_test, df_predictions, average='macro')
                prc = average_precision_score(Y_test, df_predictions, average='macro')

                # Store results
                metrics['roc'].append(roc)
                metrics['prc'].append(prc)
                metrics['fold'].append(run_idx)
                metrics["seed_set"].append(cfg_seeds)
                metrics["finetune_percentage"].append(ft_percentage)

    # Save results
    df = pd.DataFrame(metrics)
    df.to_csv('Results/experiments_paper/XGB_results_diblock_test_stoich.csv', index=False)
