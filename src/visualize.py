import math
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.express as px
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score
from torch_geometric.utils.convert import to_networkx
from typing import List
from umap import UMAP
import warnings
import wandb

warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state. Use no seed for parallelism.")


def visualize_aldeghi_results(store_pred: List, store_true: List, label: str, save_folder: str = None, epoch: int = 999, shouldPlotMetrics=False):
    assert label in ['ea', 'ip']

    xy = np.vstack([store_pred, store_true])
    z = stats.gaussian_kde(xy)(xy)

    # calculate R2 score and RMSE
    R2 = r2_score(store_true, store_pred)
    RMSE = math.sqrt(mean_squared_error(store_true, store_pred))
    if shouldPlotMetrics:
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
        wandb.log({"metrics_plot": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    return R2, RMSE


def visualize_diblock_results(store_pred: List, store_true: List, save_folder: str = None, epoch: int = 999, shouldPlotMetrics=False):
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

    # print(f"PRC = {prc_mean:.2f} +/- {prc_sem:.2f}       ROC = {roc_mean:.2f} +/- {roc_sem:.2f}")
    if shouldPlotMetrics:
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
        wandb.log({"metrics_plot": wandb.Image(fig)}, commit=False)
        plt.close()

    return prc_mean, roc_mean


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
    wandb.log({"loss_space": wandb.Image(fig)}, commit=False)
    plt.close(fig)


def visualeEmbeddingSpace(embeddings, mon_A_type, stoichiometry, model_name='', epoch=999, isFineTuning=False, should3DPlot=False, type="FT"): 
    mon_A_type = np.array(mon_A_type)
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
    
    # UMAP for 2D visualization with deterministic results
    umap_2d = UMAP(n_components=2, init='random', random_state=0) #n_components=2
    embeddings_2d = umap_2d.fit_transform(embeddings)

    # UMAP for 3D visualization if required
    if should3DPlot:
        umap_3d = UMAP(n_components=3)
        embeddings_3d = umap_3d.fit_transform(embeddings)

    df_embeddings_2d = pd.DataFrame(embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    df_embeddings_2d['Monomer A Type'] = mon_A_type
    df_embeddings_2d['Stoichiometry'] = stoichiometry

    fig_2d_monA = px.scatter(df_embeddings_2d, x='Dimension 1', y='Dimension 2', color='Monomer A Type',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                    labels={'Monomer A Type': 'Monomer A Type'},
                    title=f'2D UMAP Visualization by Monomer A Type - Epoch: {epoch}',
                    category_orders={'Monomer A Type': ['[*:1]c1cc(F)c([*:2])cc1F', '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc(-c2ccc([*:2])s2)s1', '[*:1]c1ccc([*:2])cc1', '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12', '[*:1]c1ccc([*:2])c2nsnc12', '[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2', '[*:1]c1cc2cc3sc([*:2])cc3cc2s1', '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2']},
                    opacity=0.85)
    
    # Update figure size
    fig_2d_monA.update_layout(width=800, height=600)  # Adjust the size as needed

    # Update label and axis font sizes
    fig_2d_monA.update_layout(
        title_font_size=20,
        legend_title_font_size=12,
        legend_font_size=10,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        xaxis_tickfont_size=10,
        yaxis_tickfont_size=10
    )

    # Adjust the path to save the figure
    save_folder = f'Results/{model_name}/{"FineTuningEmbeddingSpace/" if isFineTuning else "PretrainingEmbeddingSpace"}/{type}'
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure using Plotly's write_image method. Note: This requires kaleido package for static image export.
    fig_file_path = os.path.join(save_folder, f"2D_UMAP_Mon_A_{epoch}{'_FT' if isFineTuning else ''}.png")
    fig_2d_monA.write_image(fig_file_path)
    wandb.log({"2D_UMAP_Mon_A": wandb.Image(fig_file_path)}, commit=False)

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

    wandb.log({"2D_UMAP_Stoichiometry": wandb.Image(fig_file_path)}, commit=False)

    # # add pca visualization
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(embeddings)
    # df_embeddings_2d_pca = pd.DataFrame(pca_result, columns=['Dimension 1', 'Dimension 2'])
    # df_embeddings_2d_pca['Monomer A Type'] = mon_A_type
    # df_embeddings_2d_pca['Stoichiometry'] = stoichiometry


    # # use matplotlib to plot pca
    # fig, ax = plt.subplots(figsize=(7, 6))
    # for i in range(len(df_embeddings_2d_pca['Monomer A Type'].unique())):
    #     indices = np.where(df_embeddings_2d_pca['Monomer A Type'] == i)
    #     ax.scatter(df_embeddings_2d_pca.loc[indices, 'Dimension 1'], df_embeddings_2d_pca.loc[indices, 'Dimension 2'], label=f'Mon_A {i+1}')
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    # ax.legend()
    # ax.set_title('2D PCA Visualization by Monomer A Type')
    # fig.suptitle(f'PCA 2D Embeddings Colored by Monomer A Type - Epoch: {epoch}')
    # plt.savefig(os.path.join(save_folder, f"2D_PCA_Mon_A_{epoch}{'_FT' if isFineTuning else ''}.png"))
    # wandb.log({"embedding_space_monA": wandb.Image(fig)}, commit=False)
    # plt.close(fig)

    # # use matplotlib to plot pca
    # fig, ax = plt.subplots(figsize=(7, 6))
    # for i in range(len(df_embeddings_2d_pca['Stoichiometry'].unique())):
    #     indices = np.where(df_embeddings_2d_pca['Stoichiometry'] == i)
    #     ax.scatter(df_embeddings_2d_pca.loc[indices, 'Dimension 1'], df_embeddings_2d_pca.loc[indices, 'Dimension 2'], label=f'Stoichiometry {i}')
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    # ax.legend()
    # ax.set_title('2D PCA Visualization by Stoichiometry')
    # fig.suptitle(f'PCA 2D Embeddings Colored by Stoichiometry - Epoch: {epoch}')
    # plt.savefig(os.path.join(save_folder, f"2D_PCA_Stoichiometry_{epoch}{'_FT' if isFineTuning else ''}.png"))
    # wandb.log({"embedding_space_stoich": wandb.Image(fig)}, commit=False)
    # plt.close(fig)


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


def plot_subgraphs(G, subgraphs):
    # Calculate the number of rows needed to display all subgraphs with up to 3 per row
    num_rows = math.ceil(len(subgraphs) / 3)
    fig, axes = plt.subplots(num_rows, min(3, len(subgraphs)), figsize=(10, 3 * num_rows))  # Adjust size as needed

    # Flatten the axes array for easy iteration in case of a single row
    if num_rows == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()

    for ax, subgraph in zip(axes, subgraphs):
        color_map = ['orange' if node in subgraph else 'lightgrey' for node in G.nodes()]
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layouts across subplots
        nx.draw(G, pos=pos, ax=ax, with_labels=True, node_color=color_map, font_weight='bold')
        ax.set_title(f'Subgraph')

    # If there are more axes than subgraphs, hide the extra axes
    for i in range(len(subgraphs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_from_transform_attributes(data):
    # Generate the full graph and subgraphs from the transformed data
    G_full = to_networkx(data, to_undirected=True)
    G_context = G_full.subgraph(data.context_nodes_mapper.numpy())
    G_targets = [G_full.subgraph(data.target_nodes_mapper[data.target_nodes_subgraph == target_idx].numpy()) 
                for target_idx in data.target_subgraph_idxs]

    # Prepare subgraphs list including context and target subgraphs for plotting
    subgraphs = [G_context] + G_targets
    subgraph_titles = ['Context Subgraph'] + [f'Target Subgraph {i+1}' for i in range(len(G_targets))]

    # Calculate the number of subplots needed
    num_subgraphs = len(subgraphs)
    num_rows = math.ceil(num_subgraphs / 3)
    fig, axes = plt.subplots(num_rows, max(1, min(3, num_subgraphs)), figsize=(12, 4 * num_rows))
    axes = np.array(axes).flatten() if num_subgraphs > 1 else np.array([axes])

    # Generate positions for all nodes in the full graph for consistent layout
    pos = nx.spring_layout(G_full, seed=42)

    for ax, (subgraph, title) in zip(axes, zip(subgraphs, subgraph_titles)):
        # Draw the full graph in light gray as the background
        nx.draw(G_full, pos=pos, ax=ax, node_color='lightgray', edge_color='gray', alpha=0.3, with_labels=True)

        # Highlight the current subgraph
        nx.draw(subgraph, pos=pos, ax=ax, with_labels=True, node_color='orange', edge_color='black', alpha=0.7)
        ax.set_title(title)

    # Hide any unused axes
    for i in range(num_subgraphs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()