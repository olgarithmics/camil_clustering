#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/camil_clutering/camelyon_results/lipo.txt
#SBATCH --error=/home/ofourkioti/Projects/camil_clutering/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
conda activate exp_env
cd /home/ofourkioti/Projects/camil_clutering/


python run.py  --experiment_name lipo --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/SAR/feats/h5_files --label_file label_files/multi_lipo.csv --csv_files lipo_splits  --epoch 100 --save_dir SAR_Saved_model
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
