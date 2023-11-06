#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/ofourkioti/Projects/SAD_MIL/camelyon_results/cam_17_top_20.txt
#SBATCH --error=/home/ofourkioti/Projects/SAD_MIL/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
conda activate exp_env
cd /home/ofourkioti/Projects/SAD_MIL/

#python run.py --dataset camelyon --experiment_name phi --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/resnet_feats/h5_files --label_file label_files/camelyon_data.csv --csv_files camelyon_csv_files/splits_0.csv  --epoch 100 --save_dir cam_Saved_model --lambda1 0.00001
#python run_simclr.py --simclr_path  lipo_SIMCLR_checkpoints --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/simclr_imgs/h5_files/  --csv_file lipo_csv_files/splits_0.csv --simclr_batch_size 1024
for i in {0..4};
do python run.py --experiment_name cam_17_top_20  --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/sad_mil/feats/h5_files/ --label_file label_files/camelyon_data.csv --csv_file camelyon_csv_files/splits_${i}.csv --lambda1 1 --epoch 200 --eta 1 --topk 20;
done


