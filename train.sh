
#!/bin/bash
#PBS -q v100
#PBS -l nodes=1:ppn=36
#PBS -l walltime=72:00:00
#PBS -N Siamese-Monet
#PBS -A hpc_gcn03
#PBS -j oe

module purge
source activate graph
cd /work/derick/siamese-monet-project/Siamese-MoNet

singularity exec --nv -B /work,/project,/usr/lib64 /home/admin/singularity/pytorch-1.5.1-dockerhub-v4.simg python train.py -loss_dir ./results/train_results_1.json &> ./results/train_1.txt 2>&1



