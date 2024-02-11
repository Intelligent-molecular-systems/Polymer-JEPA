import numpy as np
from src.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn.functional as F

def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0, regularization=0.0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5) # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    total_loss = 0
    for data in train_loader:
        # import networkx as nx
        # graph = data[0]

        # edge_index = graph.combined_subgraphs
        # # plot 
        # G_context = nx.Graph()
        # G_context.add_edges_from(edge_index.T.cpu().numpy())
        # nx.draw(G_context, with_labels=True, node_color='skyblue')
        # import matplotlib.pyplot as plt
        # plt.show()

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

        vcRegLoss = 0
        if regularization > 0.0:
            vcRegLoss = vcReg(embeddings)

        loss = (1 - regularization) * loss + regularization * vcRegLoss

        total_loss += loss.item()        
        # Update weights of the network 
        loss.backward()
        optimizer.step()

        # Other than the target encoder, here we use exponential smoothing
        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
    
    # RISK this is different from the original code where they use arrays, idk why
    avg_trn_loss = total_loss / len(train_loader)
    return avg_trn_loss


@ torch.no_grad()
def test(loader, model, device, criterion_type=0, regularization=0.0):
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

        vcRegLoss = 0
        if regularization > 0.0:
            vcRegLoss = vcReg(embeddings)

        loss = (1 - regularization) * loss + regularization * vcRegLoss

        total_loss += loss.item()

    # RISK this is different from the original code where they use arrays, idk why
    avg_val_loss = total_loss / len(loader)
    return avg_val_loss


# vcReg = variance covariance regularization
def vcReg(embeddings, variance_threshold=1.0):
    # bring covariance matrix for each embedding to identity matrix
    # keep variance of each feature above a certain threshold via a hinge loss

    # covariance matrix
    cov = torch.cov(embeddings)
    # identity matrix
    I = torch.eye(cov.shape[0])
    # Calculate the covariance loss as the Frobenius norm of the difference between the covariance matrix and the identity matrix
    cov_loss = torch.norm(cov - I, p='fro')
    # Calculate the variance for each feature across the batch
    variances = torch.var(embeddings, dim=0)

    # Calculate the variance loss as the sum of hinge losses to ensure each variance is above the threshold
    var_loss = torch.relu(variance_threshold - variances).sum()

    return cov_loss + var_loss