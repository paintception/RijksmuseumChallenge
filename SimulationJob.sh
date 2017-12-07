#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time=04:30:00
#SBATCH --mem=100GB

module load cuDNN/5.0-CUDA-7.5.18
module load tensorflow/1.0.1-foss-2016a-Python-2.7.12-CUDA-7.5.18

python /home/s2847485/PhD/RijksmuseumChallenge/data_utils/image_processer.py

