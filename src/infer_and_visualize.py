from typing import List
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.WDNodeMPNN import WDNodeMPNN
from src.featurization_utils.featurization import poly_smiles_to_graph
import math
import os
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score


def visualize_aldeghi_results(store_pred: List, store_true: List, label: str, save_folder: str = None, epoch: int = 999):
    assert label in ['ea', 'ip']

    xy = np.vstack([store_pred, store_true])
    z = stats.gaussian_kde(xy)(xy)

    # calculate R2 score and RMSE
    R2 = r2_score(store_true, store_pred)
    RMSE = math.sqrt(mean_squared_error(store_true, store_pred))

    # now lets plot
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    plt.scatter(store_true, store_pred, s=5, c=z)
    plt.plot(np.arange(min(store_true)-0.5, max(store_true)+1.5, 1),
             np.arange(min(store_true)-0.5, max(store_true)+1.5, 1), 'r--', linewidth=1)

    plt.xlabel('True (eV)')
    plt.ylabel('Prediction (eV)')
    plt.grid()
    plt.title(f'Electron Affinity' if label == 'ea' else 'Ionization Potential')

    plt.text(min(store_true), max(store_pred), f'R2 = {R2:.3f}', fontsize=10)
    plt.text(min(store_true), max(store_pred) - 0.3, f'RMSE = {RMSE:.3f}', fontsize=10)


    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/{'EA' if label == 'ea' else 'IP'}_{epoch}.png")
    plt.close(fig)


def visualize_diblock_results(store_pred: List, store_true: List, label: str, save_folder: str = None, epoch: int = 999):
    # Convert lists to numpy arrays if they aren't already
    store_pred = np.array(store_pred)
    store_true = np.array(store_true)

    rocs = [] 
    prcs = []

    num_labels = store_true.shape[1]  # Adjust based on your true_labels' shape

    for i in range(num_labels):
        roc = roc_auc_score(store_true[:, i], store_pred[:, i], average='macro')
        prc = average_precision_score(store_true[:, i], store_pred[:, i], average='macro')
        rocs.append(roc)
        prcs.append(prc)
        
    roc_mean = np.mean(rocs)
    roc_sem = stats.sem(rocs)
    prc_mean = np.mean(prcs)
    prc_sem = stats.sem(prcs)

    print(f"PRC = {prc_mean:.2f} +/- {prc_sem:.2f}       ROC = {roc_mean:.2f} +/- {roc_sem:.2f}")
    
    # Plot the prcs results, each bar a differ color
    fig, ax = plt.subplots()
    colors = sns.color_palette('tab10')
    y_positions = np.arange(len(prcs))  # Y positions for each dot

    # Scatter plot for each class
    for i, prc in enumerate(prcs):
        ax.scatter(prc, y_positions[i], color=colors[i], s=100)  # s is the size of the dot

    # Adding error bars
    for i in range(len(prcs)):
        ax.errorbar(prcs[i], y_positions[i], xerr=prc_sem, fmt='none', ecolor='gray')

    ax.set_yticks(np.arange(len(prcs)))
    ax.set_yticklabels(['lamellar', 'cylinder', 'sphere', 'gyroid', 'disordered'])
    ax.set_xlabel('PRC')
    ax.set_title(f'PRC for each class, mean = {prc_mean:.2f} +/- {prc_sem:.2f}')
    plt.tight_layout()
    
    # Ensure save_folder exists
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f"{label}_average_auprc_epoch_{epoch}.png"))
    plt.close()