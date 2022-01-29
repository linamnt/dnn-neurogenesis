#!/bin/bash
#SBATCH --job-name=ndl-pytest
#SBATCH --time=0-00:20:00            # days-hh:mm
#
#SBATCH --cpus-per-task=6        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:t4:1            #Request GPU "generic resources"
#SBATCH --mem=32000M
#
#SBATCH --output=./output/%x-%j-%a.out     # name-jobid-arrayidx.out

# set up envrionment
SOURCEDIR=~/ndl
VENV_DIR=~/pytorch_gpu

module load python/3.6
source $VENV_DIR/bin/activate # virtual environment for project

# identify and run script
cd $SOURCEDIR
pytest 

# echo "Task ID: $SLURM_ARRAY_TASK_ID"

