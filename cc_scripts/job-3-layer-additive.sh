#!/bin/bash
#SBATCH --job-name=template
#SBATCH --array=1-10%2            # job array, start-end%max_concurrent 
#SBATCH --time=0-01:30            # days-hh:mm
#
#SBATCH --cpus-per-task=16        # Cores per GPU: Cedar - 6, Graham - 16
#SBATCH --gres=gpu:1              #Request GPU "generic resources"
#SBATCH --mem=32000M
#
#SBATCH --output=./output/%x-%j-%a.out     # name-jobid-arrayidx.out

# set up envrionment
SOURCEDIR=~/ndl/cc_scripts

module load python/3.6
source ~//bin/activate # virtual environment for project

# prepare data, usually only if using a dataloader
mkdir $SLURM_TMPDIR/data
tar -xf ~/data/data_file.tar -C $SLURM_TMPDIR/data

# identify and run script
SCRIPT=job.py # script should take as an argument the data path

python $SOURCEDIR/$SCRIPT $SLURM_TMPDIR/data

python $SOURCEDIR/$SCRIPT -t 2 -f 'combined' -d $SLURM_TMPDIR -s $SOURCEDIR

# echo "Task ID: $SLURM_ARRAY_TASK_ID"

