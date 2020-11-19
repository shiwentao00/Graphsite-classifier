"""
Generate chemical feature files for the pdb files.
"""
import os
from os import listdir
from os.path import isfile, join
import subprocess
from tqdm import tqdm


if __name__ == "__main__":
    #glosa_feature_gen = '~/Desktop/local-workspace/g-losa/glosa_v2.2/AssignChemicalFeatures'
    
    # input file list
    in_dir = '/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/all_pockets/'
    infiles = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    in_paths = []
    for f in listdir(in_dir):
        if f.endswith(".pdb"):
            in_paths.append(in_dir + f)

    # generate chemical features
    for in_path in tqdm(in_paths):
        process = subprocess.run(['java', 'AssignChemicalFeatures', in_path], 
                                cwd='/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/')
        process
    

    
    