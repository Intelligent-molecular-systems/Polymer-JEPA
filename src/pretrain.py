from contextlib import redirect_stdout
import os
import random
import string
# from PolymerJEPA_old import PolymerJEPA
from src.PolymerJEPAv2 import PolymerJEPAv2
from src.PolymerJEPA import PolymerJEPA
from src.training import train, test
from src.visualize import visualeEmbeddingSpace, visualize_loss_space
import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def pretrain(pre_trn_data, pre_val_data, transform, cfg):
    # 70-20-10 split for pretraining - validation - test data
    print(f'Pretraining training on: {len(pre_trn_data)} graphs')
    print(f'Pretraining validation on: {len(pre_val_data)} graphs')


    pre_trn_data.transform = transform
    # pre_trn_data = [x for x in pre_trn_data] # this way we can use the same transform for the validation data all the times
    pre_val_data.transform = transform
    pre_val_data = [x for x in pre_val_data] # this way we can use the same transform for the validation data all the times

    pre_trn_loader = DataLoader(dataset=pre_trn_data, batch_size=cfg.pretrain.batch_size, shuffle=True)
    pre_val_loader = DataLoader(dataset=pre_val_data, batch_size=cfg.pretrain.batch_size, shuffle=False)

    num_node_features = pre_trn_data.data_list[0].num_node_features
    num_edge_features = pre_trn_data.data_list[0].num_edge_features

    if cfg.modelVersion == 'v1':
        model = PolymerJEPA(
            nfeat_node=num_node_features,
            nfeat_edge=num_edge_features,
            nhid=cfg.model.hidden_size,
            nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
            gMHA_type=cfg.model.gMHA_type,
            rw_dim=cfg.pos_enc.rw_dim,
            pooling=cfg.model.pool,
            mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
            num_target_patches=cfg.jepa.num_targets,
            should_share_weights=cfg.pretrain.shouldShareWeights,
            regularization = cfg.pretrain.regularization,
            n_hid_wdmpnn=cfg.model.wdmpnn_hid_dim,
            shouldUse2dHyperbola=cfg.jepa.dist == 0,
            shouldLayerNorm = cfg.model.layerNorm
        ).to(cfg.device)

    elif cfg.modelVersion == 'v2':
        model = PolymerJEPAv2(
            nfeat_node=num_node_features,
            nfeat_edge=num_edge_features,
            nhid=cfg.model.hidden_size,
            rw_dim=cfg.pos_enc.rw_dim,
            pooling=cfg.model.pool,
            num_target_patches=cfg.jepa.num_targets,
            should_share_weights=cfg.pretrain.shouldShareWeights,
            regularization = cfg.pretrain.regularization,
            n_hid_wdmpnn=cfg.model.wdmpnn_hid_dim,
            shouldUse2dHyperbola=cfg.jepa.dist == 0,
            shouldLayerNorm = cfg.model.layerNorm
        ).to(cfg.device)

    else:
        raise ValueError('Invalid model version')

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
            device=cfg.device, 
            momentum_weight=next(momentum_scheduler), 
            criterion_type=cfg.jepa.dist,
            regularization=cfg.pretrain.regularization,
            inv_weight=cfg.pretrain.inv_weight, 
            var_weight=cfg.pretrain.var_weight, 
            cov_weight=cfg.pretrain.cov_weight
        )

        model.eval()

        val_loss = test(
            pre_val_loader, 
            model,
            device=cfg.device, 
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
        
        with open(f'{save_path}/hyperparams.yml', 'w') as f:
            with redirect_stdout(f): print(cfg.dump())

        scheduler.step(val_loss)

        print(f'Epoch: {epoch:03d}, Train Loss: {trn_loss:.5f}' f' Test Loss:{val_loss:.5f}')

        if epoch == 0 or epoch == cfg.pretrain.epochs - 1 or epoch % 5 == 0:

            if cfg.visualize.shouldEmbeddingSpace:
                visualeEmbeddingSpace(
                    embeddings=embedding_data[0], 
                    mon_A_type=embedding_data[1], 
                    stoichiometry=embedding_data[2],
                    model_name=model_name, 
                    epoch=epoch,
                    should3DPlot=cfg.visualize.should3DPlot
                )

            if cfg.visualize.shouldLoss:
                visualize_loss_space(
                    target_x=loss_data[0], 
                    target_y=loss_data[1],
                    model_name=model_name, 
                    epoch=epoch,
                    loss_type=cfg.jepa.dist,
                    hidden_size=cfg.model.hidden_size
                )
    
    return model, model_name