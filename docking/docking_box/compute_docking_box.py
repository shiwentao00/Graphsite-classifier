"""
Compute the docking boxes of the 14 label ligands.
"""
import os
import subprocess
import yaml


if __name__ == "__main__":
    # the tool used to compute the docking box size
    eboxsize = './eboxsize.pl'

    # ligand directory
    ligand_dir = '../../../smina/ligands/'

    # the label ligands
    ligands = [
        '2ddoA00',  #  ATP
        '1bcfE00',  #  Heme
        '1e55A00',  #  Carbonhydrate
        '5frvB00',  #  Benzene ring
        '5oy0A03',  #  Chlorophyll
        '6rfcA03',  #  lipid
        '4iu5A00',  #  Essential amino acid/citric acid/tartaric acid
        '4ineA00',  #  S-adenosyl-L-homocysteine
        '6hxiD01',  #  CoenzymeA
        '5ce8A00',  #  pyridoxal phosphate
        '5im2A00',  #  benzoic acid
        '1t57A00',  #  flavin mononucleotide
        '6frlB01',  #  morpholine ring
        '4ymzB00'   #  phosphate
    ]

    ligand_boxsize = {}
    for ligand in ligands:
        ligand_path = os.path.join(ligand_dir, ligand + '.pdbqt')
        p = subprocess.run(eboxsize + ' {}'.format(ligand_path),
            shell=True,
            stdout=subprocess.PIPE,
            text=True)
        # check=True,
        # cwd='/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/')  # working directory

        # when there is no error
        if p.returncode == 0:
            result = p.stdout
            ligand_boxsize[ligand] = float(result.strip())
        else:
            print('Something went wrong, ligand: {}'.format(ligand_path))

    with open('./docking_boxes.yaml', 'w') as f:
        yaml.dump(ligand_boxsize, f)

