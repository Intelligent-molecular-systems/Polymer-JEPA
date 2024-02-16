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
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y, embeddings, expanded_embeddings = model(data)
        all_embeddings = torch.cat((all_embeddings, embeddings))
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
    return avg_trn_loss, all_embeddings


@ torch.no_grad()
def test(loader, model, device, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        model.eval()
        target_x, target_y, _, expanded_embeddings = model(data)

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


def checkRepresentationCollapse(embeddings, model_name='', epoch=999):
    embeddings = embeddings.detach().cpu().numpy()
    means = np.mean(embeddings, axis=0)  # Mean across embedding dimensions
    stds = np.std(embeddings, axis=0)  # Standard deviation across embedding dimensions
    avg_mean = np.mean(means)  # Average mean of embeddings
    avg_std = np.mean(stds)  # Average variance of embeddings
    print(f'Average mean of embeddings: {avg_mean:.3f}, highest feat mean: {np.max(means):.3f}, lowest feat mean: {np.min(means):.3f}')
    print(f'Average std of embeddings: {avg_std:.3f}')
    
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PCA Plot
    ax1.scatter(pca_results[:, 0], pca_results[:, 1])
    ax1.set_xlabel('PCA Dimension 1')
    ax1.set_ylabel('PCA Dimension 2')
    ax1.set_title(f'PCA Visualization')

    # t-SNE Plot
    ax2.scatter(tsne_results[:, 0], tsne_results[:, 1])
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title(f't-SNE Visualization')

    fig.suptitle(f'Combined PCA and t-SNE - Avg Mean: {avg_mean:.3f} - Avg Std: {avg_std:.3f}')
    save_folder = f'Results/{model_name}'
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"Embedding_space_{epoch}.png"))
    plt.close()
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