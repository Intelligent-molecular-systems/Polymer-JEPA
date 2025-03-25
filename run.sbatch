#!/bin/sh
#SBATCH --partition=general   # Request partition. Default is 'general' 
#SBATCH --qos=medium           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=10:00:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem-per-cpu=8000     # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 


#which python 1>&2  # Write path to Python binary to standard error
#python3.8 --version   # Write Python version to standard error

#module use /opt/insy/modulefiles 
#module load cuda/12.1 miniconda/3.11 devtoolset/11
#conda activate $HOME/.local/conda/myenv

#python --version
export APPTAINER_NAME="polymer-jepa.sif"

processed_args=""

# Loop through all arguments
for arg in "$@"; do
  # Remove the leading '--' from each argument
  processed_arg="${arg#--}"
  # Append the processed argument to the string, with a space in between
  processed_args="$processed_args $processed_arg"
done

echo "$processed_args"

srun apptainer exec --bind /tudelft.net:/tudelft.net --nv $APPTAINER_NAME python main.py $processed_args
