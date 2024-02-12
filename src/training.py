import numpy as np
from src.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn.functional as F

def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    criterion = torch.nn.SmoothL1Loss(beta=0.5) # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y, embeddings = model(data)
        # Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        if regularization:
            cov_loss, var_loss = vcReg(embeddings)
            # weights: https://imbue.com/open-source/2022-04-21-vicreg/
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
    return avg_trn_loss


@ torch.no_grad()
def test(loader, model, device, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    total_loss = 0
    for data in loader:
        data = data.to(device)
        model.eval()
        target_x, target_y, embeddings = model(data)

        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        if regularization:
            cov_loss, var_loss = vcReg(embeddings)
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