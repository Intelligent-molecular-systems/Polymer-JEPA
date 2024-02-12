from typing import List
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.WDNodeMPNN import WDNodeMPNN
from src.featurization_utils.featurization import poly_smiles_to_graph
import math
import os
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, precision_recall_curve, auc


def visualize_aldeghi_results(store_pred: List, store_true: List, label: str, save_folder: str = None, epoch: int = 999):
    assert label in ['ea', 'ip']

    xy = np.vstack([store_pred, store_true])
    z = gaussian_kde(xy)(xy)

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

    # Placeholder for AUPRCs of each class
    auprcs = []

    plt.figure(figsize=(10, 7))

    num_labels = store_true.shape[1]  # Adjust based on your true_labels' shape

    for i in range(num_labels):
        precision, recall, _ = precision_recall_curve(store_true[:, i], store_pred[:, i])
        auprc = auc(recall, precision)
        auprcs.append(auprc)
        
        # Plot each class's Precision-Recall curve
        plt.plot(recall, precision, lw=2, alpha=0.3, label=f'Label {i+1} (AUPRC = {auprc:.2f})')

    # Calculate and display the average AUPRC
    average_auprc = np.mean(auprcs)
    plt.title(f'{label} Precision-Recall Curve (Average AUPRC = {average_auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.grid(True)

    # Ensure save_folder exists
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f"{label}_average_auprc_epoch_{epoch}.png"))
    plt.close()
    return average_auprc