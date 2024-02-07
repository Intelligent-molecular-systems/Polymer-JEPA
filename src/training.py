import numpy as np
from src.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn.functional as F

def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5) # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    step_losses, num_targets = [], []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y = model(data)
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

        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))
        
        # Update weights of the network 
        loss.backward()
        optimizer.step()

        # Other than the target encoder, here we use exponential smoothing
        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
        
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss # Leave none for now since maybe we'd like to return the embeddings for visualization


@ torch.no_grad()
def test(loader, model, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        target_x, target_y = model(data)

        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss