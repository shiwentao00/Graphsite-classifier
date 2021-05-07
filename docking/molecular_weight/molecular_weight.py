"""
Compute the molecular weights for the 14 docking ligands
"""
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import yaml


if __name__== "__main__":
    ligand_labels = {
        0: '2ddoA00',  # ATP
        1: '1bcfE00',  # Heme
        2: '1e55A00',  # Carbonhydrate
        3: '5frvB00',  # Benzene ring
        4: '5oy0A03',  # Chlorophyll
        5: '6rfcA03',  # lipid
        6: '4iu5A00',  # Essential amino acid/citric acid/tartaric acid
        7: '4ineA00',  # S-adenosyl-L-homocysteine
        8: '6hxiD01',  # CoenzymeA
        9: '5ce8A00',  # pyridoxal phosphate
        10: '5im2A00',  # benzoic acid
        11: '1t57A00',  # flavin mononucleotide
        12: '6frlB01',  # morpholine ring
        13: '4ymzB00'  # phosphate
    }
    ligand_dir = '../../../data/googlenet-dataset/'

    mol_weights = {}
    for i in range(14):
        ligand_name = ligand_labels[i]
        ligand_path = ligand_dir + ligand_name + '/' + ligand_name + '.sdf'
        suppl = Chem.SDMolSupplier(ligand_path)
        mols = [x for x in suppl]
        assert(len(mols) == 1)
        mol = mols[0]
        mol_weight = ExactMolWt(mol)
        mol_weights[ligand_name] = mol_weight

    with open('./ligand_weights.yaml', 'w') as f:
        yaml.dump(mol_weights, f)

