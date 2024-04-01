import wandb

def start_WB_log_hyperparameters(cfg):
    percentage = 1.0
    if cfg.finetuneDataset == 'aldeghi':
        percentage = cfg.finetune.aldeghiFTPercentage
    elif cfg.finetuneDataset == 'diblock':
        percentage = cfg.finetune.diblockFTPercentage

    tags = []
    tags.append("vicReg" if cfg.pretrain.regularization else "EMA")
    tags.append(cfg.finetuneDataset)
    tags.append(cfg.modelVersion)
    tags.append(str(percentage))

    config={
            "general": {
                "shouldPretrain": cfg.shouldPretrain,
                "shouldFinetuneOnPretrainedModel": cfg.shouldFinetuneOnPretrainedModel,
                "modelVersion": cfg.modelVersion,
                "finetuneDataset": cfg.finetuneDataset,
                "frozenWeights": cfg.frozenWeights,
            },
            "pretrain": {
                "batch_size": cfg.pretrain.batch_size,
                "epochs": cfg.pretrain.epochs,
                "lr": cfg.pretrain.lr,
                "dropout": cfg.pretrain.dropout,
                "optimizer": cfg.pretrain.optimizer,
                "regularization": cfg.pretrain.regularization,
            },
            "finetune": {   
                "isLinear": cfg.finetune.isLinear,
                "epochs": cfg.finetune.epochs,
            },
            "model": {
                "shouldUseNodeWeights": cfg.model.shouldUseNodeWeights,
                "hidden_size": cfg.model.hidden_size,
                "nlayer_gnn": cfg.model.nlayer_gnn,
                "pool": cfg.model.pool,
            },
            "pos_enc": {
                "rw_dim": cfg.pos_enc.rw_dim,
                "patch_rw_dim": cfg.pos_enc.patch_rw_dim,
            },
            "subgraphing": {
                "n_patches": cfg.subgraphing.n_patches,
                "type": cfg.subgraphing.type,
                "context_size": cfg.subgraphing.context_size,
            },
            "jepa": {
                "num_targets": cfg.jepa.num_targets,
                "dist": cfg.jepa.dist,
            }
        }

    

    if cfg.pretrain.regularization:
        config["pretrain"].update({
            "inv_weight": cfg.pretrain.inv_weight,
            "var_weight": cfg.pretrain.var_weight,
            "cov_weight": cfg.pretrain.cov_weight,
            "shouldShareWeights": cfg.pretrain.shouldShareWeights
        })

    if cfg.modelVersion == 'v1':
        config["pretrain"].update({
            "mlpmixer_dropout": cfg.pretrain.mlpmixer_dropout
        })

        config["model"].update({
            "gMHA_type": cfg.model.gMHA_type,
            "nlayer_mlpmixer": cfg.model.nlayer_mlpmixer
        })
    

    if cfg.finetuneDataset == 'aldeghi':
        config["finetune"].update({
            "aldeghiFTPercentage": cfg.finetune.aldeghiFTPercentage
        })

        config["finetune"].update({
            "property":  cfg.finetune.property
        })
       
    elif cfg.finetuneDataset == 'diblock':
        config["finetune"].update({
            "diblockFTPercentage": cfg.finetune.diblockFTPercentage
        })

    wandb.init(
        project='polymer-jepa',
        group=f'{cfg.finetuneDataset}_{cfg.modelVersion}_{percentage}',
        tags=tags,
        config=config
    )