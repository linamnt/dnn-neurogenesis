#!/bin/bash
#SBATCH --job-name=fm-test
# #SBATCH --array=1-10%2            # job array, start-end%max_concurrent 
#SBATCH --time=0-02:30            # days-hh:mm
#
#SBATCH --cpus-per-task=16        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:t4:1             #Request GPU "generic resources"
#SBATCH --mem=128000M
#
#SBATCH --output=./output/%x-%j-%a.out     # name-jobid-arrayidx.out

# set up envrionment
SOURCEDIR=~/ndl/cc_scripts
VENV_DIR=~/pytorch_gpu

module load python/3.6
source $VENV_DIR/bin/activate # virtual environment for project

# prepare data, usually only if using a dataloader
mkdir $SLURM_TMPDIR
tar -xf ~/data/fmnist.tar.gz -C $SLURM_TMPDIR

# identify and run script
SCRIPT=fm-testaccuracy.py # script should take as an argument the data path

python $SOURCEDIR/$SCRIPT -d $SLURM_TMPDIR/FashionMNIST -s $SOURCEDIR 

# echo "Task ID: $SLURM_ARRAY_TASK_ID"

