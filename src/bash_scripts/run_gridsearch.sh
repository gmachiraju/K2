source /opt/conda/etc/profile.d/conda.sh
conda activate /home/envs


script=/home/k2/K2/src/evaluation.py


python ${script} --sweep_dict --save_dir --encoder_name --gt_dir --process_args --model_args