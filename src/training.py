import matplotlib.pyplot as plt
import numpy as np
import os
from src.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn.functional as F


def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1):
    # check target and context parameters are the same (weight sharing)
    # for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
    #     if not torch.equal(param_q, param_k):
    #         print(f"context and target are not same.")
    #     else:
    #         print(f"context == target !!")

    total_loss = 0
    all_embeddings = torch.tensor([], device=device)
    mon_A_type = torch.tensor([], device=device)
    stoichiometry = []
    inv_losses = []
    cov_losses = []
    var_losses = []
    target_x_saved = None
    target_y_saved = None
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y, expanded_embeddings = model(data)
        if i == 0:
            target_x_saved = target_x
            target_y_saved = target_y
        if i % 5 == 0: # around 6k if training on 35/40k
            embeddings = model.encode(data).detach()
            mon_A_type = torch.cat((mon_A_type, data.mon_A_type.detach()), dim=0)
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
            inv_losses.append(loss.item())
            cov_losses.append(cov_loss.item())
            var_losses.append(var_loss.item())  
            
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
    if regularization:
        avg_inv_loss = np.mean(inv_losses)
        avg_cov_loss = np.mean(cov_losses)
        avg_var_loss = np.mean(var_losses)
        print(f'Average Inverse Loss: {avg_inv_loss:.5f}, Average Covariance Loss: {avg_cov_loss:.5f}, Average Variance Loss: {avg_var_loss:.5f}')        
        print(f'weighted values: inv_loss: {inv_weight*avg_inv_loss:.5f}, cov_loss: {cov_weight*avg_cov_loss:.5f}, var_loss: {var_weight*avg_var_loss:.5f}')

    visualize_embedding_data = (all_embeddings, mon_A_type, stoichiometry)
    visualize_hyperbola_data = (target_x_saved, target_y_saved)
    return avg_trn_loss, visualize_embedding_data, visualize_hyperbola_data


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

# import networkx as nx
        # graph = data[0]

        # edge_index = graph.combined_subgraphs
        # # plot 
        # G_context = nx.Graph()
        # G_context.add_edges_from(edge_index.T.cpu().numpy())
        # nx.draw(G_context, with_labels=True, node_color='skyblue')
        # import matplotlib.pyplot as plt
        # plt.show()