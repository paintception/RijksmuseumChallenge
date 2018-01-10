#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --time=72:00:00
#SBATCH --mem=100GB

module load cuDNN/5.0-CUDA-7.5.18
module load tensorflow/1.0.1-foss-2016a-Python-2.7.12-CUDA-7.5.18

cat /home/s2847485/PhD/RijksmuseumChallenge/StyleTransfer/experiments/layers_of_interest.txt | while read LINE
do
	python /home/s2847485/PhD/RijksmuseumChallenge/StyleTransfer/experiments/style_transfer.py $LINE
done

