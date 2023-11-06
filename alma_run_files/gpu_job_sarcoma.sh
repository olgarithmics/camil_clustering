#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/graph_CHARM/sar_results/k_2_3_4_8_graph_layer.txt
#SBATCH --error=/home/ofourkioti/Projects/graph_CHARM/sar_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate exp_env
cd /home/ofourkioti/Projects/graph_CHARM/

python run.py  --k 2 3 4 8 --dataset sarcoma --experiment_name k_2_3_4_8_graph_layer --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/tf_feats_256/resnet_feats/h5_files --label_file label_files/sarcoma_data.csv --csv_files lipo_csv_files  --epoch 100 --save_dir SAR_Saved_model
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
