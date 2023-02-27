#!/bin/bash

### For GPU support submit this script with: sbatch --gres=gpu:1 submit_gpu_python.sh

#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6000

# Load and activate Anaconda3 module
module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate;

# Load additional modules
# module load ...

# Activate User Environment
conda activate ml_env

# Run python script
srun --gres=gpu:1 -n1 --exclusive python /hpc/users/keyvan.mahjoory/prj_eegatt/codes/workflow_cross_validation_new.py