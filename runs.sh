#!/bin/sh
for seeds in 0
do
    for diblockFTPercentage in 0.04 0.24 0.48 0.8
    do
        #layer_norm=1
        #early_stopping=0
        use_augmented_data=0
        #sbatch ./run.sbatch --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --use_augmented_data ${use_augmented_data} --finetune.aldeghiFTPercentage ${aldeghiFTPercentage}
        layer_norm=0
        early_stopping=0
        sbatch ./run.sbatch --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --use_augmented_data ${use_augmented_data} --finetune.diblockFTPercentage ${diblockFTPercentage}
    done
done
# Run this with config:, PL true&false, Pretraining True and False, 3 different seeds (repetitions)Va