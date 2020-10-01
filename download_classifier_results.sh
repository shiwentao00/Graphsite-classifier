run=11

mkdir ./results/classifier_run_${run}/

cd ./results/classifier_run_${run}/

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/Siamese-MoNet/results/*${run}* ./

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/trained_models/trained_classifier_model_${run}.pt ../../../trained_models/

cd ../../


