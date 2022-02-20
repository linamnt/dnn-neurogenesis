#!/bin/bash
#SBATCH --job-name=compare_best
#SBATCH --time=2-08:30            # days-hh:mm
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
mkdir $SLURM_TMPDIR/data
tar -xf ~/data/cifar10.tar -C $SLURM_TMPDIR/data

# identify and run script
SCRIPT=c10-compare-best.py # script should take as an argument the data path

python $SOURCEDIR/$SCRIPT -d $SLURM_TMPDIR/data -s $SOURCEDIR -n 250 -e 14
echo date
# echo "Task ID: $SLURM_ARRAY_TASK_ID"

