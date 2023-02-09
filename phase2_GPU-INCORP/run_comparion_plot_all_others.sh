#!/bin/bash

#SBATCH --job-name=GPU-INCORP_KN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=knni@ucdavis.edu
#SBATCH --output=/home/knni/git/phase2_GPU-INCORP/scripts/logs/%j.out
#SBATCH --error=/home/knni/git/phase2_GPU-INCORP/scripts/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres gpu:1
#SBATCH --mem=8G
#SBATCH --time=30:00:00

python run_plot_sample_ba.py
python run_plot_sample_nb.py
python run_plot_sample_svc.py
python run_plot_feature_ba.py
python run_plot_feature_nb.py
python run_plot_feature_svc.py
