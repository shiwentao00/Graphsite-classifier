#!/bin/bash
# Software tool: ~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py
# use this Python: ~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/bin/pythonsh
# ligands directory: ~/Desktop/local-workspace/siamese-monet-project/baseline_data/14_class_pdb/
# output directory: ~/Desktop/local-workspace/siamese-monet-project/vina/ligand-pdbqt/

ligands=~/Desktop/local-workspace/siamese-monet-project/baseline_data/14_class_pdb
python=~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/bin/pythonsh
code=~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py
out=~/Desktop/local-workspace/siamese-monet-project/vina/ligand-pdbqt/
for ligand in $ligands/*; do
    echo $ligand
    out_file=${ligand##*/}
    out_file=${out_file::-4}
    out_file="$out_file.pdbqt"
    out_file="$out$out_file"
    echo ${out_file}
    $python $code -l $ligand -A 'hydrogens_bonds' -o $out_file
done