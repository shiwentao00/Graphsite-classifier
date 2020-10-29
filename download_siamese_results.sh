run=68

mkdir ./siamese_results/run_${run}/

cd ./siamese_results/run_${run}/

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/Siamese-MoNet/siamese_results/*${run}* ./

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/trained_models/*${run}* ../../../trained_models/

cd ../../../embeddings/

mkdir ./run_${run}/

rsync -a derick@smic.hpc.lsu.edu:/work/derick/siamese-monet-project/embeddings/run_${run}/* ./run_${run}/

cd ../Siamese-MoNet/


