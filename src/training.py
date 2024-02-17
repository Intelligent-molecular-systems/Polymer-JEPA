import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn.functional as F


def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    total_loss = 0
    all_embeddings = torch.tensor([], device=device)
    mon_A_type = torch.tensor([], device=device)
    stoichiometry = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y, expanded_embeddings = model(data)
        embeddings = model.encode(data)
        mon_A_type = torch.cat((mon_A_type, data.mon_A_type), dim=0)
        all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
        stoichiometry.extend(data.stoichiometry)
        # Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
        if criterion_type == 0:
            criterion = torch.nn.SmoothL1Loss(beta=0.5) # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        if regularization:
            cov_loss, var_loss = vcReg(expanded_embeddings)
            inv_weight = inv_weight
            var_weight = var_weight
            cov_weight = cov_weight
        
            # vicReg objective
            loss = inv_weight * loss + var_weight * var_loss + cov_weight * cov_loss

        total_loss += loss.item()        
        # Update weights of the network 
        loss.backward()
        optimizer.step()

        if not regularization: # if not vicReg, use EMA
            with torch.no_grad():
                for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
        
    
    # RISK this is different from the original code where they use arrays, idk why
    avg_trn_loss = total_loss / len(train_loader)
    visualize_info = (all_embeddings, mon_A_type, stoichiometry)
    return avg_trn_loss, visualize_info


@ torch.no_grad()
def test(loader, model, device, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        model.eval()
        target_x, target_y, expanded_embeddings = model(data)

        if criterion_type == 0:
            criterion = torch.nn.SmoothL1Loss(beta=0.5)
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        if regularization:
            cov_loss, var_loss = vcReg(expanded_embeddings)
            # vicReg objective
            loss = inv_weight * loss + var_weight * var_loss + cov_weight * cov_loss

        total_loss += loss.item()

    # RISK this is different from the original code where they use arrays, idk why
    avg_val_loss = total_loss / len(loader)
    return avg_val_loss




def vcReg(embeddings):
    def off_diagonal(x):
        # Create a mask that is 0 on the diagonal and 1 everywhere else
        n = x.size(0)
        mask = torch.ones_like(x) - torch.eye(n, device=x.device)
        return x * mask

    N = embeddings.size(0)
    D = embeddings.size(1)
    
    # Center the embeddings
    embeddings_centered = embeddings - embeddings.mean(dim=0)
    
    # Covariance matrix calculation for centered embeddings
    cov = (embeddings_centered.T @ embeddings_centered) / (N - 1)
    
    # Covariance loss focusing only on off-diagonal elements
    cov_loss = off_diagonal(cov).pow(2).sum() / D
    
    # Variance loss calculation
    var = torch.var(embeddings, dim=0) + 1e-04
    std_devs = torch.sqrt(var)
    std_loss = torch.mean(F.relu(1 - std_devs))
    
    return cov_loss, std_loss


def checkRepresentationCollapse(embeddings, mon_A_type, stoichiometry, model_name='', epoch=999): 
    embeddings = embeddings.detach().cpu().numpy()
    means = np.mean(embeddings, axis=0)  # Mean across embedding dimensions
    stds = np.std(embeddings, axis=0)  # Standard deviation across embedding dimensions
    avg_mean = np.mean(means)  # Average mean of embeddings
    avg_std = np.mean(stds)  # Average variance of embeddings
    print(f'Average mean of embeddings: {avg_mean:.3f}, highest feat mean: {np.max(means):.3f}, lowest feat mean: {np.min(means):.3f}')
    print(f'Average std of embeddings: {avg_std:.3f}')

    # randomly sample 4000 embeddings for plotting so that its easier to visualize and faster to compute
    if len(embeddings) > 4000:
        indices = np.random.choice(len(embeddings), 2000, replace=False)
        embeddings = embeddings[indices]
        mon_A_type = mon_A_type[indices]
        stoichiometry = np.array(stoichiometry)[indices]

    # Dimensionality reduction
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    num_classes = 9
    colors_monA = plt.cm.get_cmap('tab10', num_classes)
    colors_stoch = plt.cm.get_cmap('viridis', 3)  # 3 stoichiometry classes

    save_folder = f'Results/{model_name}'
    os.makedirs(save_folder, exist_ok=True)

    # Plot colored by mon_A_type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i in range(num_classes):
        indices = np.where(mon_A_type == i)
        axes[0].scatter(pca_results[indices, 0], pca_results[indices, 1], color=colors_monA(i), label=f'Mon_A {i+1}')
        axes[1].scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors_monA(i), label=f'Mon_A {i+1}')
    for ax in axes:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
    axes[0].set_title('PCA Visualization')
    axes[1].set_title('t-SNE Visualization')
    fig.suptitle(f' PCA and t-SNE Embeddings Colored by Monomer A type - Avg Mean: {avg_mean:.3f} - Avg Std: {avg_std:.3f}')
    plt.savefig(os.path.join(save_folder, f"Embedding_mon_A_{epoch}.png"))
    plt.close(fig)
    
    stoichiometry = np.array(stoichiometry)
    # Plot colored by stoichiometry classification
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    stoichiometries = ["1:1", "3:1", "1:3"]
    for i, stoch in enumerate(stoichiometries):  # 3 stoichiometry classes
        indices = np.where(stoichiometry == stoch)[0]
        axes[0].scatter(pca_results[indices, 0], pca_results[indices, 1], color=colors_stoch(i), label=f'Stoich Class {stoch}')
        axes[1].scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors_stoch(i), label=f'Stoich Class {stoch}')
    for ax in axes:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
    axes[0].set_title('PCA Visualization')
    axes[1].set_title('t-SNE Visualization')
    fig.suptitle(f' PCA and t-SNE Embeddings Colored by Stoichiometry - Avg Mean: {avg_mean:.3f} - Avg Std: {avg_std:.3f}')
    plt.savefig(os.path.join(save_folder, f"Embedding_stoichiometry_{epoch}.png"))
    plt.close(fig)
#plot

# import networkx as nx
        # graph = data[0]

        # edge_index = graph.combined_subgraphs
        # # plot 
        # G_context = nx.Graph()
        # G_context.add_edges_from(edge_index.T.cpu().numpy())
        # nx.draw(G_context, with_labels=True, node_color='skyblue')
        # import matplotlib.pyplot as plt
        # plt.show()