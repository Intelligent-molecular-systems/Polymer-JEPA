from contextlib import redirect_stdout
import os
import random
import string
# from PolymerJEPA_old import PolymerJEPA
from src.JEPA_models.PolymerJEPAv2 import PolymerJEPAv2
from src.JEPA_models.PolymerJEPA import PolymerJEPA
from src.JEPA_models.GeneralJEPA import GeneralJEPAv1
from src.JEPA_models.GeneralJEPAv2 import GeneralJEPAv2
from src.training import train, test, reset_parameters
from src.visualize import visualeEmbeddingSpace, visualize_loss_space
import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def pretrain(pre_trn_data, pre_val_data, cfg, device):
    print(f'Pretraining training on: {len(pre_trn_data)} graphs')
    print(f'Pretraining validation on: {len(pre_val_data)} graphs')

    pre_trn_loader = DataLoader(dataset=pre_trn_data, batch_size=cfg.pretrain.batch_size, shuffle=True, num_workers=cfg.num_workers)
    pre_val_loader = DataLoader(dataset=pre_val_data, batch_size=cfg.pretrain.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if cfg.finetuneDataset == 'aldeghi' or cfg.finetuneDataset == 'diblock':
        if cfg.modelVersion == 'v1':
            model = PolymerJEPA(
                nfeat_node=133,
                nfeat_edge=14,
                nhid=cfg.model.hidden_size,
                nlayer_gnn=cfg.model.nlayer_gnn,
                nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                gMHA_type=cfg.model.gMHA_type,
                rw_dim=cfg.pos_enc.rw_dim,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                pooling=cfg.model.pool,
                n_patches=cfg.subgraphing.n_patches,
                mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization=cfg.pretrain.regularization,
                shouldUse2dHyperbola=cfg.jepa.dist == 0,
                shouldUseNodeWeights=cfg.model.shouldUseNodeWeights
            ).to(device)

        elif cfg.modelVersion == 'v2':
            model = PolymerJEPAv2(
                nfeat_node=133,
                nfeat_edge=14,
                nhid=cfg.model.hidden_size,
                nlayer_gnn=cfg.model.nlayer_gnn,
                rw_dim=cfg.pos_enc.rw_dim,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                pooling=cfg.model.pool,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization=cfg.pretrain.regularization,
                shouldUse2dHyperbola=cfg.jepa.dist == 0,
                shouldUseNodeWeights=cfg.model.shouldUseNodeWeights,
            ).to(device)

        else:
            raise ValueError('Invalid model version')
    
    elif cfg.finetuneDataset == 'zinc':
        if cfg.modelVersion == 'v1':
            model = GeneralJEPAv1(
                nfeat_node=28,
                nfeat_edge=4,
                nhid=cfg.model.hidden_size,
                nlayer_gnn=cfg.model.nlayer_gnn,
                nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                gMHA_type=cfg.model.gMHA_type,
                rw_dim=cfg.pos_enc.rw_dim,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                pooling=cfg.model.pool,
                n_patches=cfg.subgraphing.n_patches,
                mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization=cfg.pretrain.regularization,
                shouldUse2dHyperbola=cfg.jepa.dist == 0,
                shouldUseNodeWeights=cfg.model.shouldUseNodeWeights
            ).to(device)

        elif cfg.modelVersion == 'v2':
            model = GeneralJEPAv2(
                nfeat_node=28,
                nfeat_edge=4,
                nhid=cfg.model.hidden_size,
                nlayer_gnn=cfg.model.nlayer_gnn,
                rw_dim=cfg.pos_enc.rw_dim,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                pooling=cfg.model.pool,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization=cfg.pretrain.regularization,
                shouldUse2dHyperbola=cfg.jepa.dist == 0,
                shouldUseNodeWeights=cfg.model.shouldUseNodeWeights
            ).to(device)
        else:
            raise ValueError('Invalid model version')


    # print('model', model)
    reset_parameters(model)
    print(f"\nNumber of parameters: {count_parameters(model)}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.pretrain.lr, 
        weight_decay=cfg.pretrain.wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=cfg.pretrain.lr_decay,
        patience=cfg.pretrain.lr_patience,
        verbose=True
    )

    # Create EMA scheduler for target encoder param update
    ipe = len(pre_trn_loader)
    ema_params = [0.996, 1.0]
    momentum_scheduler = (ema_params[0] + i*(ema_params[1]-ema_params[0])/(ipe*cfg.pretrain.epochs)
                        for i in range(int(ipe*cfg.pretrain.epochs)+1))


    random.seed(time.time())
    model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    print(f"Model name: {model_name}")
    
    # Pretraining
    for epoch in tqdm(range(cfg.pretrain.epochs), desc='Pretraining Epochs'):
        model.train()
        trn_loss, embedding_data, loss_data = train(
            pre_trn_loader, 
            model, 
            optimizer, 
            device=device, 
            momentum_weight=next(momentum_scheduler), 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.pretrain.regularization,
            inv_weight=cfg.pretrain.inv_weight, 
            var_weight=cfg.pretrain.var_weight, 
            cov_weight=cfg.pretrain.cov_weight,
            epoch=epoch,
            dataset=cfg.finetuneDataset
        )

        model.eval()

        val_loss = test(
            pre_val_loader, 
            model,
            device=device, 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.pretrain.regularization,
            inv_weight=cfg.pretrain.inv_weight, 
            var_weight=cfg.pretrain.var_weight, 
            cov_weight=cfg.pretrain.cov_weight
        )

        # save model weights at each epoch
        save_path = f'Models/Pretrain/{model_name}'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f'{save_path}/model.pt')
        
        if epoch == 0:
            with open(f'{save_path}/hyperparams.yml', 'w') as f:
                with redirect_stdout(f): print(cfg.dump())

        scheduler.step(val_loss)

        print(f'Epoch: {epoch:03d}, Train Loss: {trn_loss:.5f}' f' Test Loss:{val_loss:.5f}')

        if epoch == 0 or epoch == cfg.pretrain.epochs - 1 or epoch % 5 == 0:

            if cfg.visualize.shouldEmbeddingSpace and cfg.finetuneDataset == 'aldeghi':
                # model v2 does not have initial context and target embeddings, since there is no initial encoder
                if cfg.modelVersion == 'v1':
                    # visualize initial context embeddings (wdmpnn output)
                    visualeEmbeddingSpace(
                        embeddings=embedding_data[0], 
                        mon_A_type=embedding_data[-2], 
                        stoichiometry=embedding_data[-1],
                        model_name=model_name, 
                        epoch=epoch,
                        should3DPlot=cfg.visualize.should3DPlot,
                        type="context_wdmpnn_output"
                    )

                    # visualize initial target embeddings (wdmpnn output)
                    visualeEmbeddingSpace(
                        embeddings=embedding_data[1], 
                        mon_A_type=embedding_data[-2], 
                        stoichiometry=embedding_data[-1],
                        model_name=model_name, 
                        epoch=epoch,
                        should3DPlot=cfg.visualize.should3DPlot,
                        type="target_wdmpnn_output"
                    )
                
                # visualize target embeddings (output of the target encoder for the full graph)
                visualeEmbeddingSpace(
                    embeddings=embedding_data[2], 
                    mon_A_type=embedding_data[-2], 
                    stoichiometry=embedding_data[-1],
                    model_name=model_name, 
                    epoch=epoch,
                    should3DPlot=cfg.visualize.should3DPlot,
                    type="target_encoder_output"
                )  
                
                # visualize graph embeddings (output of the target encoder for the full graph)
                visualeEmbeddingSpace(
                    embeddings=embedding_data[3], 
                    mon_A_type=embedding_data[-2], 
                    stoichiometry=embedding_data[-1],
                    model_name=model_name, 
                    epoch=epoch,
                    should3DPlot=cfg.visualize.should3DPlot,
                    type="target_encoder_full_graph_output"
                ) 

                # visualize context encoder embeddings
                visualeEmbeddingSpace(
                    embeddings=embedding_data[4], 
                    mon_A_type=embedding_data[-2], 
                    stoichiometry=embedding_data[-1],
                    model_name=model_name, 
                    epoch=epoch,
                    should3DPlot=cfg.visualize.should3DPlot,
                    type="context_encoder_output"
                )

            if cfg.visualize.shouldLoss:
                visualize_loss_space(
                    target_embeddings=loss_data[0], 
                    predicted_target_embeddings=loss_data[1],
                    model_name=model_name, 
                    epoch=epoch,
                    loss_type=cfg.jepa.dist,
                    hidden_size=cfg.model.hidden_size
                )
    
    return model, model_name


def count_parameters(model):
    # For counting number of parameteres: need to remove unnecessary DiscreteEncoder, and other additional unused params
    return sum(p.numel() for p in model.parameters() if p.requires_grad)