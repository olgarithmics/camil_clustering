#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/camil_clustering/results/cam_17_camil_sb.txt
#SBATCH --error=/home/ofourkioti/Projects/camil_clustering/results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
conda activate exp_env
cd /home/ofourkioti/Projects/camil_clustering/

for i in {0..3};
do
python run.py  --experiment_name cam_17_camil_sb --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/feats/h5_files/ --label_file label_files/camelyon_17.csv --csv_file camelyon_17_splits/splits_${i}.csv   --k_sample 8 --single_branch;
done
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/images/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/patches/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/






