#!/bin/bash
# Software tool: ~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py
# use this Python: ~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/bin/pythonsh
# pockets directory: ~/Desktop/local-workspace/siamese-monet-project/vina/original-pockets
# output directory: ~/Desktop/local-workspace/siamese-monet-project/vina/pocket-pdbqt/

pockets=~/Desktop/local-workspace/siamese-monet-project/vina/original-pockets
python=~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/bin/pythonsh
code=~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py
out=~/Desktop/local-workspace/siamese-monet-project/vina/pocket-pdbqt/
for pocket in $pockets/*; do
    # echo $pocket
    out_file=${pocket##*/}
    out_file=${out_file::-4}
    out_file="$out_file.pdbqt"
    out_file="$out$out_file"
    # echo ${out_file}
    #$python $code -r $pocket -A 'bonds_hydrogens' -o $out_file
    $python $code -r $pocket -A 'hydrogens' -o $out_file
done