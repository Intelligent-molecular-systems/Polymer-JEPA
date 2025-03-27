#!/bin/sh
#for augmented_fraction in 0.0 0.1 0.2 0.4 0.6 0.8 1.0
#do
#sbatch ./run.sbatch --augmented_data_fraction ${augmented_fraction} 
#done
layer_norm=1
sbatch ./run.sbatch --pretrain.layer_norm ${layer_norm}
#layer_norm=0
#sbatch ./run.sbatch --pretrain.layer_norm ${layer_norm}

