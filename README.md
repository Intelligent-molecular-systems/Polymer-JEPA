# Polymer-JEPA

```
conda create polymer-jepa python=3.10
conda activate polymer-jepa
pip install torch-geometric==2.3.0 torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch==2.1 wandb networkx yacs metis kaleido tensorboard einops pillow tqdm pandas rdkit matplotlib plotly seaborn umap 
```

- Run with: `python main.py`

- Setup the training settings from `src.config` or via command line arguments, i.e.:

```
python main.py shouldPretrain False
```