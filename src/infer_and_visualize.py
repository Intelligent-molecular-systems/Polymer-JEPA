from typing import List
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.WDNodeMPNN import WDNodeMPNN
from src.featurization import poly_smiles_to_graph
import math
import os
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error


def infer_by_dataloader(loader, model, device):
    out, ea, ip = [], [], []

    model.eval()
    model.to(device)

    for batch in tqdm(loader, total=len(loader), desc="processing batches"):
        out += model(batch.to(device)).cpu().tolist()
        ea += batch.y_EA.tolist()
        ip += batch.y_IP.tolist()

    return out, ea, ip


def visualize_results(store_pred: List, store_true: List, label: str, save_folder: str = None, epoch: int = 999):
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


    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/{'EA' if label == 'ea' else 'IP'}_{epoch}.png")

    return fig, R2, RMSE


# def infer(smiles_list: List[str], ea_list: List[float], ip_list: List[float], model, device = None, visualize=False) -> np.ndarray:
#     assert type(model) == str or type(model) == WDNodeMPNN

#     if device is None:
#         device = torch.device("cpu")

#     graphs = []
#     valid_indexes = []

#     N = range(len(smiles_list))
#     for i, smiles, ea, ip  in tqdm(zip(N, smiles_list, ea_list, ip_list), total=len(N), desc="creating graphs"):
#         try:
#             graph = poly_smiles_to_graph(smiles, [ea], [ip])
#             graphs.append(graph)
#             valid_indexes.append(i)

#         except Exception as e:
#             print(f"{smiles} is not valid polymer string")

#     if type(model) == str:
#         model_temp = WDNodeMPNN(133, 14, hidden_dim=300)
#         model_temp.load_state_dict(torch.load(model))
#         model = model_temp
#         model.to(device)

#     loader = DataLoader(dataset=graphs,
#                batch_size=16, shuffle=False)

#     out, ea, ip = infer_by_dataloader(loader, model, device)

#     print(f"{len(smiles_list) - len(valid_indexes)} failed entries (!)")


#     if visualize:
#         visualize_results(out, ea, label='ea', save_folder='Results/figs')
#         visualize_results(out, ip, label='ip', save_folder='Results/figs')

#     pred = np.full(len(smiles_list), None, dtype=object)
#     pred[valid_indexes] = out

#     return pred    