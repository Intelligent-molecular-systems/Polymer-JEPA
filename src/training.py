import numpy as np
from src.JEPA_models.model_utils.hyperbolic_dist import hyperbolic_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def train(train_loader, model, optimizer, device, momentum_weight,sharp=None, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1, epoch=0, dataset='aldeghi', jepa_weight=0.5, m_w_weight=0.5):
    
    total_loss = 0

    # initialize lists for visualization
    all_graph_embeddings, all_initial_context_embeddings, all_initial_target_embeddings, all_target_encoder_embeddings, all_context_encoder_embeddings = torch.tensor([], requires_grad=False, device=device), torch.tensor([], requires_grad=False, device=device), torch.tensor([], requires_grad=False, device=device), torch.tensor([], requires_grad=False, device=device), torch.tensor([], requires_grad=False, device=device)
    mon_A_type, stoichiometry, chain_arch, inv_losses, cov_losses, var_losses = [], [], [], [], [], []
    target_embeddings_saved, predicted_target_embeddings_saved = None, None
  
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        target_embeddings, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings, initial_context_embeddings, initial_target_embeddings,  context_embeddings, target_encoder_embeddings, graph_embeddings, pseudoLabelPrediction = model(data)
        
        if dataset == 'aldeghi':
            ### visualization ###
            if i == 0: # save the first target_x and target_y for visualization of loss space
                target_embeddings_saved = target_embeddings
                predicted_target_embeddings_saved = predicted_target_embeddings
            if i % 6 == 0: # around 6k if training on 35/40k, save the embeddings for visualization of embedding space
                all_graph_embeddings = torch.cat((all_graph_embeddings, graph_embeddings.detach().clone()), dim=0)
                all_initial_context_embeddings = torch.cat((all_initial_context_embeddings, initial_context_embeddings.detach().clone()), dim=0)
                all_initial_target_embeddings = torch.cat((all_initial_target_embeddings, initial_target_embeddings.detach().clone()), dim=0)
                all_target_encoder_embeddings = torch.cat((all_target_encoder_embeddings, target_encoder_embeddings.detach().clone()), dim=0)
                all_context_encoder_embeddings = torch.cat((all_context_encoder_embeddings, context_embeddings.detach().clone()), dim=0)
                mon_A_type.extend(data.mon_A_type)
                stoichiometry.extend(data.stoichiometry)

                for graph in data.to_data_list():
                    chain_arch.append(str(graph.full_input_string.split("|")[-1]).split(":")[1])
                
            ### End visualization ### 
            
        # Distance function: 0 = 2d Hyper, 1 = Euclidean, 2 = Hyperbolic
        if criterion_type == 0:
            inv_loss = F.smooth_l1_loss(predicted_target_embeddings, target_embeddings) # https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html
        elif criterion_type == 1:
            jepa_loss = F.mse_loss(predicted_target_embeddings, target_embeddings) * jepa_weight
            # normalize the M_ensemble
            if m_w_weight > 0:
                data.M_ensemble = (data.M_ensemble - torch.mean(data.M_ensemble)) / torch.std(data.M_ensemble)
                pseudolabel_loss = F.mse_loss(pseudoLabelPrediction.squeeze(1), data.M_ensemble.float()) * m_w_weight
                inv_loss = jepa_loss + pseudolabel_loss
            else:
                inv_loss = jepa_loss
        elif criterion_type == 2:
            inv_loss = hyperbolic_dist(predicted_target_embeddings, target_embeddings)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        wandb_log_dict = {"pretrn_trn_inv_loss": inv_loss.item()}

        if regularization: # if vicReg is used
            context_cov_loss, context_var_loss = vcReg(expanded_context_embeddings)  
            target_cov_loss, target_var_loss = vcReg(expanded_target_embeddings)
            cov_loss = context_cov_loss + target_cov_loss
            var_loss = context_var_loss + target_var_loss
            
            wandb_log_dict["pretrn_trn_cov_loss"] = cov_loss.item()
            wandb_log_dict["pretrn_trn_var_loss"] = var_loss.item()

            inv_losses.append(inv_loss.item())
            cov_losses.append(cov_loss.item())
            var_losses.append(var_loss.item())  
            
            # vicReg objective
            loss = inv_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss
            wandb_log_dict["pretrn_trn_total_loss"] = loss.item()
        else:
            loss = inv_loss
            

        total_loss += loss.item()        

        loss.backward()
        optimizer.step()

        if not regularization: # if not vicReg, use EMA
            with torch.no_grad():
                for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
                 # Update normalization layer weights and biases as well with EMA
                if model.layer_norm:
                    for norm_q, norm_k in zip([model.context_norm], [model.target_norm]):
                        norm_k.weight.data.mul_(momentum_weight).add_((1. - momentum_weight) * norm_q.weight.detach().data)
                        norm_k.bias.data.mul_(momentum_weight).add_((1. - momentum_weight) * norm_q.bias.detach().data)
        
    
    avg_trn_loss = total_loss / len(train_loader)
    if regularization:
        avg_inv_loss = np.mean(inv_losses)
        avg_cov_loss = np.mean(cov_losses)
        avg_var_loss = np.mean(var_losses)
        print(f'\ninv_loss: {avg_inv_loss:.5f}, cov_loss: {avg_cov_loss:.5f}, var_loss: {avg_var_loss:.5f}')        
        print(f'Weighted values: inv_loss: {inv_weight*avg_inv_loss:.5f}, cov_loss: {cov_weight*avg_cov_loss:.5f}, var_loss: {var_weight*avg_var_loss:.5f}\n')

    embeddings_data = (all_initial_context_embeddings, all_initial_target_embeddings, all_target_encoder_embeddings, all_graph_embeddings, all_context_encoder_embeddings, mon_A_type, stoichiometry, chain_arch)

    loss_data = (target_embeddings_saved, predicted_target_embeddings_saved)
    wandb_log_dict["pretrain_epoch"] = epoch
    wandb.log(wandb_log_dict)
    return avg_trn_loss, embeddings_data, loss_data


@ torch.no_grad()
def test(loader, model, device, criterion_type=0, regularization=False, inv_weight=25, var_weight=25, cov_weight=1, jepa_weight=0.5, m_w_weight=0.5):
    total_loss = 0
    for idx, data in enumerate(loader):
        data = data.to(device)
        model.eval()
        target_embeddings, predicted_target_embeddings, expanded_context_embeddings, expanded_target_embeddings, _, _, _, _, _, pseudoLabelPrediction = model(data)

        if criterion_type == 0:
            inv_loss = F.smooth_l1_loss(predicted_target_embeddings, target_embeddings)
        elif criterion_type == 1:
            jepa_loss = F.mse_loss(predicted_target_embeddings, target_embeddings) * jepa_weight
            if m_w_weight > 0:
                data.M_ensemble = (data.M_ensemble - torch.mean(data.M_ensemble)) / torch.std(data.M_ensemble)
                pseudolabel_loss = F.mse_loss(pseudoLabelPrediction.squeeze(1), data.M_ensemble.float()) * m_w_weight
                if idx == 0:
                    print(f'jepa_loss: {jepa_loss.item()}, pseudolabel_loss: {pseudolabel_loss.item()}')
                inv_loss = jepa_loss + pseudolabel_loss
            else:
                inv_loss = jepa_loss
        elif criterion_type == 2:
            inv_loss = hyperbolic_dist(predicted_target_embeddings, target_embeddings)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        wandb_log_dict = {"pretrn_val_inv_loss": inv_loss.item()}

        if regularization:
            context_cov_loss, context_var_loss = vcReg(expanded_context_embeddings)  
            target_cov_loss, target_var_loss = vcReg(expanded_target_embeddings)
            cov_loss = context_cov_loss + target_cov_loss
            var_loss = context_var_loss + target_var_loss
           
            wandb_log_dict["pretrn_val_cov_loss"] = cov_loss.item()
            wandb_log_dict["pretrn_val_var_loss"] = var_loss.item()
            
            loss = inv_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss
            wandb_log_dict["pretrn_val_total_loss"] = loss.item()
        else:
            loss = inv_loss

        total_loss += loss.item()

    avg_val_loss = total_loss / len(loader)
    wandb.log(wandb_log_dict)
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
    cov_loss = off_diagonal(cov).pow_(2).sum() / D
    
    # Variance loss calculation
    std_devs = torch.sqrt(embeddings.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_devs))
    
    return cov_loss, std_loss


def reset_parameters(module):
    def reset_module_parameters(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0)  # Initialize bias to zero
            
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else: # Fallback for modules without a `reset_parameters` method
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    for child in module.children():
        reset_module_parameters(child)

