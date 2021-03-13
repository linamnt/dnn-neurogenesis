#!/bin/bash
#SBATCH --job-name=sgdVSadam
#SBATCH --time=0-03:30            # days-hh:mm
#
#SBATCH --cpus-per-task=16        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:t4:1             #Request GPU "generic resources"
#SBATCH --mem=64000M
#
#SBATCH --output=./output/%x-%j.out     # name-jobid-arrayidx.out

# set up envrionment
SOURCEDIR=~/ndl/cc_scripts
VENV_DIR=~/pytorch_gpu

module load python/3.6
source $VENV_DIR/bin/activate # virtual environment for project

# prepare data, usually only if using a dataloader
mkdir $SLURM_TMPDIR/data
tar -xf ~/data/mnist.tar -C $SLURM_TMPDIR/data

# identify and run script
SCRIPT=mnist-hpt-adamsgd.py # script should take as an argument the data path

# run the script
python $SOURCEDIR/$SCRIPT -d /home/linatran/ndl/ -s $SOURCEDIR -n 250 

