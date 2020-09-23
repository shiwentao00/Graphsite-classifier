run=62

mkdir ./results/run_${run}/

cd ./results/run_${run}/

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/Siamese-MoNet/results/*${run}* ./

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/trained_models/*${run}* ../../../trained_models/

cd ../../../embeddings/

mkdir ./run_${run}/

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/embeddings/run_${run}/* ./run_${run}/

cd ../Siamese-MoNet/


