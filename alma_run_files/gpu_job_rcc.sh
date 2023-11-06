#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/SAD_MIL/camelyon_results/camil_rcc_dense.txt
#SBATCH --error=/home/ofourkioti/Projects/SAD_MIL/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
conda activate exp_env
cd /home/ofourkioti/Projects/SAD_MIL/

for i in {0..3};
do
python run.py  --experiment_name camil_rcc --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/feats/h5_files/ --label_file label_files/rcc_data.csv --adj_shape 8 --csv_file rcc_file_splits/splits_${i}.csv  --lambda1 1 --epoch 200 --eta 1 --topk 60 --subtyping --n_classes 3;
done
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/images/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon17/patches/
#/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/






