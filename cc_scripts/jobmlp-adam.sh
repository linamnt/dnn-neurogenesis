#!/bin/bash
#SBATCH --job-name=mlp-adam
#SBATCH --time=0-08:00            # days-hh:mm
#SBATCH --cpus-per-task=16        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:t4:1              #Request GPU "generic resources"
#SBATCH --mem=16000M
#
#SBATCH --output=./output/%x-%j.out     # name-jobid-arrayidx.out


# set up envrionment
SOURCEDIR=~/ndl/cc_scripts
VENV=~/pytorch_gpu

module load python/3.6
source $VENV/bin/activate # virtual environment for project

# prepare data, usually only if using a dataloader
tar -xf ~/data/mnist.tar -C $SLURM_TMPDIR

# identify and run script
SCRIPT=mnist-hpt-adam.py # script should take as an argument the data path

# Run the script
python $SOURCEDIR/$SCRIPT -d $SLURM_TMPDIR -s $SOURCEDIR -n 250


