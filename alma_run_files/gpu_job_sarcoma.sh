#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/camil_clustering/results/lipo_5.txt
#SBATCH --error=/home/ofourkioti/Projects/camil_clustering/results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
conda activate exp_env
cd /home/ofourkioti/Projects/camil_clustering/


for i in {0..4};
do python run.py  --experiment_name lipo_5 --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/SAR/feats/h5_files --label_file label_files/multi_lipo.csv --csv_file lipo_csv_splits/splits_${i}.csv  --epoch 100 --save_dir SAR_Saved_model --lr 0.005;
done

