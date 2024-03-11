import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score
from typing import List
from umap import UMAP
import warnings

warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state. Use no seed for parallelism.")


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

    # v1 or v2 diblock or aldeghi FT percentage
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/{'EA' if label == 'ea' else 'IP'}_{epoch}.png")
    plt.close(fig)


def visualize_diblock_results(store_pred: List, store_true: List, save_folder: str = None, epoch: int = 999):
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
        plt.savefig(os.path.join(save_folder, f"average_auprc_epoch_{epoch}.png"))
    plt.close()


def visualize_loss_space(target_embeddings, predicted_target_embeddings, model_name='', epoch=999, loss_type=0, hidden_size=128):
    target_embeddings = target_embeddings.reshape(-1, 2).detach().clone().cpu().numpy()
    predicted_target_embeddings = predicted_target_embeddings.reshape(-1, 2).detach().clone().cpu().numpy()
  
    # Unpack the points: convert lists of tuples to separate lists for x and y coordinates
    x_x, x_y = zip(*target_embeddings)  # Unpack target_x points
    y_x, y_y = zip(*predicted_target_embeddings)  # Unpack target_y points

    # Create a figure and a set of subplots
    fig = plt.figure(figsize=(12, 5))

    # Plot for target_x
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.scatter(x_x, x_y, color='blue', label='Target X')
    plt.title('True coordinates from target encoder')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    # Plot for target_y
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    # Generate data for the hyperbola (Q=1 case)
    if loss_type == 0:
        x_min, x_max = np.min(target_embeddings), np.max(target_embeddings)
        x_vals = np.linspace(max(1, x_min), x_max, 400)
        y_vals = np.sqrt(x_vals**2 - 1)
        plt.plot(x_vals, y_vals, color='blue', linestyle='-', linewidth=2)

    plt.scatter(y_x, y_y, color='red', label='Target Y')
    plt.title('Predicted coordinates from context and predictor network')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    # Show plot
    plt.tight_layout()
    save_folder = f'Results/{model_name}/PretrainingLossSpace'
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{epoch}.png"))
    plt.close(fig)


def visualeEmbeddingSpace(embeddings, mon_A_type, stoichiometry, model_name='', epoch=999, isFineTuning=False, should3DPlot=False, type="FT"): 
    mon_A_type = mon_A_type.cpu().numpy()
    stoichiometry = np.array(stoichiometry)

    if isFineTuning:
        embeddings = embeddings.detach().clone().cpu().numpy()
    else:
        embeddings = embeddings.cpu().clone().numpy()

    # Calculate mean and standard deviation statistics
    means = np.mean(embeddings, axis=0)
    stds = np.std(embeddings, axis=0)
    avg_mean = np.mean(means)
    avg_std = np.mean(stds)
    print(f'\n***{type}***\nAverage mean of embeddings: {avg_mean:.3f}, highest feat mean: {np.max(means):.3f}, lowest feat mean: {np.min(means):.3f}')
    print(f'Average std of embeddings: {avg_std:.3f}\n')

    # Randomly sample embeddings for easier visualization and faster computation
    desired_size = 3500
    if len(embeddings) > desired_size:
        indices = np.random.choice(len(embeddings), desired_size, replace=False)
        embeddings = embeddings[indices]
        mon_A_type = mon_A_type[indices]
        stoichiometry = stoichiometry[indices]
    
    mon_A_type = mon_A_type + 1  # Shift to 1-indexing for better visualization

    # UMAP for 2D visualization with deterministic results
    # why to use UMAP: https://stats.stackexchange.com/questions/402668/intuitive-explanation-of-how-umap-works-compared-to-t-sne
    umap_2d = UMAP(n_components=2, init='random', random_state=0) #n_components=2
    embeddings_2d = umap_2d.fit_transform(embeddings)

    # UMAP for 3D visualization if required
    if should3DPlot:
        umap_3d = UMAP(n_components=3)
        embeddings_3d = umap_3d.fit_transform(embeddings)

    # save_folder = f'Results/{model_name}/{"FineTuningEmbeddingSpace" if isFineTuning else "PretrainingEmbeddingSpace"}'
    # os.makedirs(save_folder, exist_ok=True)

    # Define color maps for visualization
    # num_classes = 9
    # colors_monA = plt.cm.get_cmap('tab10', num_classes)
    # colors_stoch = plt.cm.get_cmap('viridis', 3)  # Assuming 3 stoichiometry classes

    # 2D Visualization colored by Monomer A Type
    # fig, ax = plt.subplots(figsize=(7, 6))
    # for i in range(num_classes):
    #     indices = np.where(mon_A_type == i)
    #     ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], color=colors_monA(i), label=f'Mon_A {i+1}')
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    # ax.legend()
    # ax.set_title('2D UMAP Visualization by Monomer A Type')
    # fig.suptitle(f'UMAP 2D Embeddings Colored by Monomer A Type - Epoch: {epoch}')
    # plt.savefig(os.path.join(save_folder, f"2D_UMAP_mon_A_{epoch}{'_FT' if isFineTuning else ''}.png"))
    # plt.close(fig)
    df_embeddings_2d = pd.DataFrame(embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df_embeddings_2d['Monomer A Type'] = pd.Categorical(mon_A_type) 
    df_embeddings_2d['Stoichiometry'] = stoichiometry
    mon_A_types_sorted = sorted(df_embeddings_2d['Monomer A Type'].unique())

    fig_2d_monA = px.scatter(df_embeddings_2d, x='Dimension 1', y='Dimension 2', color='Monomer A Type',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                    labels={'Monomer A Type': 'Monomer A Type'},
                    title=f'2D UMAP Visualization by Monomer A Type - Epoch: {epoch}',
                    category_orders={'Monomer A Type': mon_A_types_sorted},
                    opacity=0.85)

    # Adjust the path to save the figure
    save_folder = f'Results/{model_name}/{"FineTuningEmbeddingSpace/" if isFineTuning else "PretrainingEmbeddingSpace"}/{type}'
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure using Plotly's write_image method. Note: This requires kaleido package for static image export.
    fig_file_path = os.path.join(save_folder, f"2D_UMAP_Mon_A_{epoch}{'_FT' if isFineTuning else ''}.png")
    fig_2d_monA.write_image(fig_file_path)

    # 2D Visualization colored by Stoichiometry
    # fig, ax = plt.subplots(figsize=(7, 6))
    # stoichiometries = ["1:1", "3:1", "1:3"]
    # for i, stoch in enumerate(stoichiometries):
    #     indices = np.where(stoichiometry == stoch)  # Update this if stoichiometry is not numeric
    #     ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], color=colors_stoch(i), label=f'Stoichiometry {stoch}')
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    # ax.legend()
    # ax.set_title('2D UMAP Visualization by Stoichiometry')
    # fig.suptitle(f'UMAP 2D Embeddings Colored by Stoichiometry - Epoch: {epoch}')
    # plt.savefig(os.path.join(save_folder, f"2D_Embedding_stoichiometry_{epoch}{'_FT' if isFineTuning else ''}.png"))
    # plt.close(fig)
    fig_2d_stoich = px.scatter(df_embeddings_2d, x='Dimension 1', y='Dimension 2', color='Stoichiometry',
                           color_discrete_sequence=px.colors.qualitative.Set1, 
                           labels={'Stoichiometry': 'Stoichiometry'},
                           title=f'2D UMAP Visualization by Stoichiometry - Epoch: {epoch}',
                           category_orders={'Stoichiometry': ['1:1', '3:1', '1:3']},
                           opacity=0.85)

    # Adjust the path to save the figure
    save_folder = f'Results/{model_name}/{"FineTuningEmbeddingSpace" if isFineTuning else "PretrainingEmbeddingSpace"}/{type}'
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure using Plotly's write_image method. Note: This requires the `kaleido` package for static image export.
    fig_file_path = os.path.join(save_folder, f"2D_UMAP_Stoichiometry_{epoch}{'_FT' if isFineTuning else ''}.png")
    fig_2d_stoich.write_image(fig_file_path)
   

    # def mpl_to_plotly_cmap(cmap, num_classes):
    #     colors = cmap(np.linspace(0, 1, num_classes))
    #     plotly_colors = ['rgb' + str((int(color[0]*255), int(color[1]*255), int(color[2]*255))) for color in colors]
    #     return plotly_colors

    # if should3DPlot:
    #     # 3D Visualization for Monomer A Type with Plotly
    #     monA_colors_plotly = mpl_to_plotly_cmap(plt.cm.tab10, num_classes)
    #     fig = px.scatter_3d(
    #         x=tsne_results_3d[:, 0], y=tsne_results_3d[:, 1], z=tsne_results_3d[:, 2],
    #         color=mon_A_type.astype(str),  # Ensure discrete coloring
    #         color_discrete_sequence=monA_colors_plotly,
    #         title=f'3D t-SNE Visualization Colored by Monomer A Type - Epoch: {epoch}',
    #         labels={'color': 'Monomer A Type'}
    #     )
    #     fig.write_html(os.path.join(save_folder, f"3D_tsne_mon_A_type_{epoch}{'_FT' if isFineTuning else ''}.html"))

    #     # 3D Visualization for Stoichiometry with Plotly
    #     stoich_colors_plotly = mpl_to_plotly_cmap(plt.cm.viridis, len(stoichiometries))
    #     fig = px.scatter_3d(
    #         x=tsne_results_3d[:, 0], y=tsne_results_3d[:, 1], z=tsne_results_3d[:, 2],
    #         color=stoichiometry,  # Ensure discrete coloring
    #         color_discrete_sequence=stoich_colors_plotly,
    #         title=f'3D t-SNE Visualization Colored by Stoichiometry - Epoch: {epoch}',
    #         labels={'color': 'Stoichiometry'}
    #     )
    #     fig.write_html(os.path.join(save_folder, f"3D_tsne_stoichiometry_{epoch}{'_FT' if isFineTuning else ''}.html"))