#!/bin/bash
#PBS -q v100
#PBS -l nodes=1:ppn=36
#PBS -l walltime=72:00:00
#PBS -A hpc_michal01
#PBS -j oe

run=25

module purge
source activate graph
cd /work/derick/siamese-monet-project/Siamese-MoNet

singularity exec --nv -B /work,/project,/usr/lib64 /home/admin/singularity/pytorch-1.5.1-dockerhub-v4-python38 python train_classifier.py &> ./results/train_classifier_${run}.txt 2>&1



