#!/bin/sh

#SBATCH --partition=general   # Request partition. Default is 'general' 
#SBATCH --qos=medium           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=7:00:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem-per-cpu=5GB     # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=NONE      # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=./log/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=./log/slurm_%j.err  # Set name of error log. %j is the Slurm jobId


#which python 1>&2  # Write path to Python binary to standard error
#python3.8 --version   # Write Python version to standard error

#module use /opt/insy/modulefiles 
#module load cuda/12.1 miniconda/3.11 devtoolset/11
#conda activate $HOME/.local/conda/thesis

#python --version

processed_args=""

# Loop through all arguments
for arg in "$@"; do
  # Remove the leading '--' from each argument
  processed_arg="${arg#--}"
  # Append the processed argument to the string, with a space in between
  processed_args="$processed_args $processed_arg"
done

echo "$processed_args"
srun python main.py $processed_args