# Polymer-JEPA


### Setup enviroment
```
conda create polymer-jepa python=3.10
conda activate polymer-jepa
pip install torch-geometric==2.3.0 torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch==2.1 wandb networkx yacs metis kaleido tensorboard einops pillow tqdm pandas rdkit matplotlib plotly seaborn umap 
```
Alternatively setup the environment using containerization (see `environment_apptainer.yml` and `apptainer_env.def` files).

### Run
- Run with: `python main.py`
- Setup the training settings from `src.config` or via command line arguments. See also an example in `runs.sh` and `run.sbatch`. 

```
python main.py shouldPretrain False
```

### Experiments
Besides running experiments with JEPA pretraining, we also run the experiments using a baseline random forest model, as implemented in [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop/tree/master/chemprop/features), and using an input-space SSL architecture developed for the same architecture [[1]](#1).
We provide the results in the folder Results/experiments_paper/. The code for running the random forest model is located in the scripts `src.train_test_rf.py`
 

### Credits 
The JEPA models code and `transform.py` are largely based on [Graph-JEPA](https://github.com/geriskenderi/graph-jepa) code, itself based on the [Graph-ViT-MLPMixer](https://github.com/XiaoxinHe/Graph-ViT-MLPMixer) code.
The featurization process code is taken from [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop/tree/master/chemprop/features) code. 



### References
<a id="1">[1]</a> 
Gao, Q., Dukker, T., Schweidtmann, A. M., & Weber, J. M. (2024). 
Self-supervised graph neural networks for polymer property prediction. Molecular Systems Design & Engineering, 9(11), 1130-1143.