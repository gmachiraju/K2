#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --account=hazy
#SBATCH --partition=hazy
#SBATCH --job-name=test8
#SBATCH --mem=25G

# source /scr/biggest/gmachi/miniconda3/etc/profile.d
conda activate /scr/biggest/gmachi/miniconda3/envs/kkenv

script=/scr/gmachi/prospection/K2/src/test_signal.py

python ${script}
