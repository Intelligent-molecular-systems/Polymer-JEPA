import random
from src.config import cfg
from src.data import create_data
from src.finetune import finetune
from src.PolymerJEPA import PolymerJEPA
from src.PolymerJEPAv2 import PolymerJEPAv2
from src.pretrain import pretrain
import torch


def run():
    # [RISK]: how to handle the dataset? i am not sure from a dataset instance i can slice it like this
    # https://github.com/pyg-team/pytorch_geometric/issues/4223 
    aldeghi_dataset, transform = create_data(cfg)
    
    # pretraning always done on the aldeghi dataset since its bigger dataset and no issues with homopolymer or tri, penta...blocks polymers
    # which would require different subgraphing techniques

    pre_data = aldeghi_dataset[:int(cfg.pretrain.pretrainPercentage*len(aldeghi_dataset))].copy()
    if cfg.finetuneDataset == 'aldeghi':
        print('Finetuning will be on aldeghi dataset...')
        ft_data = aldeghi_dataset[int(cfg.pretrain.pretrainPercentage*len(aldeghi_dataset)):].copy()
    elif cfg.finetuneDataset == 'diblock':
        print('Loading diblock dataset for finetuning...')
        graphs = torch.load('Data/diblock_graphs_list.pt')
        random.seed(12345)
        graphs = random.sample(graphs, len(graphs))
        ft_data = graphs
    else:
        raise ValueError('Invalid dataset name')

    if cfg.shouldPretrain:
        model, model_name = pretrain(pre_data, transform, cfg)
    else:
        # load model from finetuning
        model_name = 'ZspxRWsm'
        if cfg.modelVersion == 'v1':
            model = PolymerJEPA(
                nfeat_node=aldeghi_dataset.data_list[0].num_node_features,
                nfeat_edge=aldeghi_dataset.data_list[0].num_edge_features,
                nhid=cfg.model.hidden_size,
                nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                gMHA_type=cfg.model.gMHA_type,
                rw_dim=cfg.pos_enc.rw_dim,
                pooling=cfg.model.pool,
                mlpmixer_dropout=cfg.pretrain.mlpmixer_dropout,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization = cfg.pretrain.regularization
            ).to(cfg.device)

        elif cfg.modelVersion == 'v2':
            model = PolymerJEPAv2(
                nfeat_node=aldeghi_dataset.data_list[0].num_node_features,
                nfeat_edge=aldeghi_dataset.data_list[0].num_edge_features,
                nhid=cfg.model.hidden_size,
                rw_dim=cfg.pos_enc.rw_dim,
                pooling=cfg.model.pool,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                num_target_patches=cfg.jepa.num_targets,
                should_share_weights=cfg.pretrain.shouldShareWeights,
                regularization = cfg.pretrain.regularization
            ).to(cfg.device)

        else:
            raise ValueError('Invalid model version')

        model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}.pt', map_location=cfg.device))
    

    if cfg.shouldFinetune:
        finetune(ft_data, transform, model, model_name, cfg)
    


        
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
