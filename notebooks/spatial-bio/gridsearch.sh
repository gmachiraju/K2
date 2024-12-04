#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --account=hazy
#SBATCH --partition=hazy
#SBATCH --job-name=grid20
#SBATCH --mem=35G

# source /scr/biggest/gmachi/miniconda3/etc/profile.d
conda activate /scr/biggest/gmachi/miniconda3/envs/kkenv

python /scr/gmachi/prospection/K2/notebooks/spatial-bio/run_gridsearch.py




