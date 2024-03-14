import random
from src.config import cfg
from src.data import create_data
from src.finetune import finetune
# from PolymerJEPA_old import PolymerJEPA
from src.PolymerJEPAv2 import PolymerJEPAv2
from src.PolymerJEPA import PolymerJEPA
from src.pretrain import pretrain
from src.training import reset_parameters
import string
import time
import torch


def run():
    pretrn_dataset, ft_dataset, val_dataset = create_data(cfg)
    pretrn_dataset.shuffle()
    # ft_dataset.shuffle()
    # pretraning always done on the aldeghi dataset since its bigger dataset and no issues with homopolymer or tri, penta...blocks polymers
    # which would require different subgraphing techniques

    if cfg.finetuneDataset == 'aldeghi':
        print('Finetuning will be on aldeghi dataset...')
        ft_data = ft_dataset[:int(cfg.pretrain.aldeghiFTPercentage*len(ft_dataset))]
        # for official result use the full val_dataset, but to run experiments fast i can use 0.95
        pre_test_data = val_dataset # .copy()
        ft_test_data = val_dataset #.copy()


    elif cfg.finetuneDataset == 'diblock':
        # for official result use the full val_dataset, but to run experiments fast i can use 0.95
        pre_test_data = val_dataset # .copy()
        print('Loading diblock dataset for finetuning...')
        graphs = torch.load('Data/diblock_graphs_list.pt')
        random.seed(12345)
        graphs = random.sample(graphs, len(graphs))
        ft_data = graphs[:int(cfg.pretrain.diblockFTPercentage*len(graphs))]
        ft_test_data = graphs[int(cfg.pretrain.diblockFTPercentage*len(graphs)):]
    else:
        raise ValueError('Invalid dataset name')

    model_name = None

    if cfg.shouldPretrain:
        model, model_name = pretrain(pretrn_dataset, pre_test_data, cfg)

    if cfg.shouldFinetune:
        if cfg.modelVersion == 'v1':
            model = PolymerJEPA(
                nfeat_node=pretrn_dataset.data_list[0].num_node_features,
                nfeat_edge=pretrn_dataset.data_list[0].num_edge_features,
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
                shouldUseNodeWeights=True
            ).to(cfg.device)

        elif cfg.modelVersion == 'v2':
            model = PolymerJEPAv2(
                nfeat_node=pretrn_dataset.data_list[0].num_node_features,
                nfeat_edge=pretrn_dataset.data_list[0].num_edge_features,
                nhid=cfg.model.hidden_size,
                nlayer_gnn=cfg.model.nlayer_gnn,
                rw_dim=cfg.pos_enc.rw_dim,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                pooling=cfg.model.pool,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization=cfg.pretrain.regularization,
                shouldUse2dHyperbola=cfg.jepa.dist == 0,
                shouldUseNodeWeights=True
            ).to(cfg.device)

        else:
            raise ValueError('Invalid model version')

        if cfg.shouldFinetuneOnPretrainedModel:
            if not model_name: # it means we are not pretraining in the current run
                model_name = 'wqtEG48z'

            model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}/model.pt', map_location=cfg.device))
        
            finetune(ft_data, ft_test_data, model, model_name, cfg)
        
        else:
            reset_parameters(model)
            # in case we are not finetuning on a pretrained model
            random.seed(time.time())
            model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            model_name += '_NotPretrained'
            finetune(ft_data, ft_test_data, model, model_name, cfg)

if __name__ == '__main__':
    run()




# check features dimensions for batches
# for i in range(0, len(dataset), 20):
#     try:
#         Batch.from_data_list(dataset[i:i+20])
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         for data in dataset[i:i+20]:
#             print(data)
#             quit()


# params_before = {name: param.clone() for name, param in model.named_parameters()}
# params_after = {name: param.clone() for name, param in model.named_parameters()}

            # Compare parameters
            # for name, param_before in params_before.items():
            #     param_after = params_after[name]
            #     # Check if the same (using torch.equal to compare tensors)
            #     if not torch.equal(param_before, param_after):
            #         print(f"Parameter {name} has changed.")
            #     else:
            #         print(f"Parameter {name} remains the same.")