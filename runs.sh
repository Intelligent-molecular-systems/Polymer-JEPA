#!/bin/sh
for augmented_fraction in 0.0 0.1 0.2 0.4 0.6 0.8 1.0
do
sbatch ./run.sbatch --augmented_data_fraction ${augmented_fraction} 
done
#sbatch ./run.sbatch