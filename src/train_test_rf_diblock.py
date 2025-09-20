#!/usr/bin/env python
# Modified script from https://github.com/coleygroup/polymer-chemprop-data/blob/main/results/vipea/train_test_rf.py
# The pickle data files are copied from https://github.com/coleygroup/polymer-chemprop-data/tree/main/datasets/diblock-phases/rf_inputs
import collections
import random
import numpy as np
import pandas as pd
from src.config import cfg, update_cfg

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

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
    metrics_test = collections.defaultdict(list)
    for cfg_seeds, seeds in seed_sets.items():
        print("Used seeds:", seeds)

        for ft_percentage in diblockFTPercentages:
            print(f"Running for fine-tune percentage: {ft_percentage}")

            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=12345)
            phase1_labels = Y_full.idxmax(axis=1).tolist()
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

                # Train RF
                model = RandomForestClassifier(n_estimators=100, max_depth=None, criterion='gini', class_weight="balanced",
                                       random_state=42, n_jobs=12)
                # train model
                model.fit(X_train_subset, Y_train_subset)

                # make predictions
                Y_pred = model.predict_proba(X_test)  # n_labels x n_samples x n_classes

                # check order of classes in array to make sure we're getting the right probability
                for c in model.classes_:
                    if len(c) == 2:
                        assert c[1] == 1
                    else:
                        assert c[0] == 0

                _Y_pred = []
                for y_pred in Y_pred:
                    if y_pred.shape[1] == 2:
                        _Y_pred.append(y_pred)
                    elif y_pred.shape[1] == 1:
                        y_pred = np.pad(y_pred, [(0,0),(0,1)], mode='constant', constant_values=0.)
                        _Y_pred.append(y_pred)

                df_predictions = pd.DataFrame(np.array(_Y_pred)[:,:,1].T, columns=Y_test.columns)

                # Assuming Y_test is a DataFrame like the original Y_full
                # and Y_pred is a list of arrays per class from predict_proba

                roc = roc_auc_score(Y_test, df_predictions, average='macro')
                # average precision is the same as the area under the precision-recall curve
                prc = average_precision_score(Y_test, df_predictions, average='macro')

                # Store results
                metrics['prc'].append(prc)
                metrics['fold'].append(run_idx)
                metrics["seed_set"].append(cfg_seeds)
                metrics["finetune_percentage"].append(ft_percentage)
    df = pd.DataFrame(metrics)
    df.to_csv(f'Results/experiments_paper/RF_results_diblock_test_stoich.csv', index=False)
