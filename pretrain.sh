#!/bin/sh

sbatch ./run.sh --modelVersion v1 --subgraphing.context_size 0.3
sbatch ./run.sh --modelVersion v1 --subgraphing.context_size 0.5
sbatch ./run.sh --modelVersion v1 --subgraphing.context_size 0.7
sbatch ./run.sh --modelVersion v1 --subgraphing.context_size 0.9