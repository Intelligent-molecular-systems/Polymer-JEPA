import random
from src.config import cfg
from src.data import create_data
from src.finetune import finetune
from src.PolymerJEPA import PolymerJEPA
from src.PolymerJEPAv2 import PolymerJEPAv2
from src.pretrain import pretrain
import string
import time
import torch


def run():
    # [RISK]: how to handle the dataset? i am not sure from a dataset instance i can slice it like this
    # https://github.com/pyg-team/pytorch_geometric/issues/4223 
    aldeghi_dataset, transform = create_data(cfg)
    
    # pretraning always done on the aldeghi dataset since its bigger dataset and no issues with homopolymer or tri, penta...blocks polymers
    # which would require different subgraphing techniques

    if cfg.finetuneDataset == 'aldeghi':
        print('Finetuning will be on aldeghi dataset...')
        pre_data = aldeghi_dataset[:int(cfg.pretrain.pretrainPercentage*len(aldeghi_dataset))].copy()
        ft_data = aldeghi_dataset[int(cfg.pretrain.pretrainPercentage*len(aldeghi_dataset)):].copy()

    elif cfg.finetuneDataset == 'diblock':
        pre_data = aldeghi_dataset # we can use the full dataset for pretraining
        print('Loading diblock dataset for finetuning...')
        graphs = torch.load('Data/diblock_graphs_list.pt')
        random.seed(12345)
        graphs = random.sample(graphs, len(graphs))
        # ft_data = graphs
        ft_data = graphs[:int(0.2*len(graphs))]
    else:
        raise ValueError('Invalid dataset name')

    model_name = None

    if cfg.shouldPretrain:
        model, model_name = pretrain(pre_data, transform, cfg)

    if cfg.shouldFinetune:
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


        if cfg.shouldFinetuneOnPretrainedModel:
            if not model_name: # it means we are not pretraining in the current run
                model_name = 'Y66tN9Wz'
            # To print the model parameters after loading
            # params_before = {name: param.clone() for name, param in model.named_parameters()}

            model.load_state_dict(torch.load(f'Models/Pretrain/{model_name}.pt', map_location=cfg.device))
            # params_after = {name: param.clone() for name, param in model.named_parameters()}

            # Compare parameters
            # for name, param_before in params_before.items():
            #     param_after = params_after[name]
            #     # Check if the same (using torch.equal to compare tensors)
            #     if not torch.equal(param_before, param_after):
            #         print(f"Parameter {name} has changed.")
            #     else:
            #         print(f"Parameter {name} remains the same.")

            fine_tuned_model = finetune(ft_data, transform, model, model_name, cfg)
            # params_after_finetune = {name: param for name, param in fine_tuned_model.named_parameters()}
            # # Compare parameters
            # for name, param_after in params_after.items():
            #     param_after_finetune = params_after_finetune[name]
            #     # Check if the same (using torch.equal to compare tensors)
            #     if not torch.equal(param_after, param_after_finetune):
            #         print(f"Parameter {name} has changed.")
            #     else:
            #         print(f"Parameter {name} remains the same.")
        else:
            # in case we are not finetuning on a pretrained model
            random.seed(time.time())
            model_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            model_name += '_NotPretrained'
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
