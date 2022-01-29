#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --time=0-23:00            # days-hh:mm
#SBATCH --cpus-per-task=16        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:t4:1              #Request GPU "generic resources"
#SBATCH --mem=32000M
#
#SBATCH --output=./output/%x-%j.out     # name-jobid-arrayidx.out


# set up envrionment
SOURCEDIR=~/ndl/cc_scripts
VENV=~/pytorch_gpu

module load python/3.6
source $VENV/bin/activate # virtual environment for project

# prepare data, usually only if using a dataloader
mkdir $SLURM_TMPDIR/data
tar -xf ~/data/cifar10.tar -C $SLURM_TMPDIR/data

# identify and run script
SCRIPT=c10-ablation.py # script should take as an argument the data path

# Run the script
python $SOURCEDIR/$SCRIPT -d $SLURM_TMPDIR -s $SOURCEDIR -n 250 -t True


