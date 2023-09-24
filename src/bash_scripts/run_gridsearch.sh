source /opt/conda/etc/profile.d/conda.sh
conda activate /home/envs

# Before running, please follow the instructions below:
# 1. Please change fields in job_params.py

# 2. Change these fields
#------------------------
encoder_name="tile2vec"
#------------------------

script="/home/k2/K2/src/evaluation.py"
gt_dir="/home/data/tinycam/train/gt_graphs_"${encoder_name}
save_dir="/home/k2/K2/src/outputs/"${encoder_name}"_gridsearch"

python ${script} --save_dir ${save_dir} --encoder_name ${encoder_name} --gt_dir ${gt_dir}