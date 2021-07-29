"""For each class, set a label ligand, then compute the
Tanimoto coefficients between each ligand in the dataset and the label ligand.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import MACCSkeys
import rdkit.Chem as Chem
from tqdm import tqdm
import yaml
import argparse

# for user-defined packages
import sys
sys.path.append('../')
from gnn.dataloader import read_cluster_file_from_yaml
from gnn.dataloader import merge_clusters


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-cls',
                        required=True,
                        type=int,
                        help='which class to find pairs.')

    parser.add_argument('-out',
                        required=False,
                        default='../../data/class_tanimoto_with_label_ligand/',
                        help='directory for output.')

    parser.add_argument('-tc_threshold',
                        default=0.7,
                        required=False,
                        type=float,
                        help='threshold of TC to calculate the ratio')

    return parser.parse_args()


def compute_similarity(fp_a, fp_b):
    """compute the similarity between two smiles fingerprints"""
    return FingerprintSimilarity(fp_a, fp_b)


label_ligands = {
    # ATP
    0: 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N'
}


if __name__ == "__main__":
    args = get_args()
    cls = int(args.cls)
    out = args.out
    tc_threshold = float(args.tc_threshold)

    # load the 14 classes
    with open('../gnn/train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cluster_file_dir = config['cluster_file_dir']
    merge_info = config['merge_info']
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # load the pocket-smile dictionary
    with open('../data/pocket-smiles.yaml') as f:
        smiles_dict = yaml.load(f, Loader=yaml.FullLoader)

    # pre-compute finger prints and store in hashmap
    print('computing finger prints...')
    fps = {}
    total, success = 0, 0
    cluster = clusters[cls]
    for pocket in tqdm(cluster):
        total += 1
        if pocket in smiles_dict:
            smiles = smiles_dict[pocket]
            mol = Chem.MolFromSmiles(smiles)
            fps[pocket] = MACCSkeys.GenMACCSKeys(mol)
            success += 1
    success_rate = success / total
    print(f'success rate: {success_rate}')

    # label ligand
    label_ligand = label_ligands[cls]
    mol_label = Chem.MolFromSmiles(label_ligand)
    fp_label = MACCSkeys.GenMACCSKeys(mol_label)

    # compute the Tanimoto coefficients
    total, success = 0, 0
    tc, similar = [], []
    print('computing tanimoto similarities...')
    for i, pocket in enumerate(tqdm(cluster)):
        total += 1
        if pocket in fps:
            fp_pocket = fps[pocket]
            if fp_label and fp_pocket:
                s = compute_similarity(fp_pocket, fp_label)
                tc.append(s)
                if s >= tc_threshold:
                    similar.append(s)
                success += 1
    success_rate = success / total
    print(f'success rate: {success_rate}')

    # plot histogram
    tc = np.array(tc)
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(tc, 10, density=False)
    ax.set_xlabel('Tanimoto similarity with label')
    ax.set_ylabel('Number of ligands')
    ax.set_title(f'class_{cls}')
    fig.tight_layout()
    plt.savefig(os.path.join(out, f'class_{cls}.png'))

    # compute ratio
    ratio = len(similar) / total
    print(f'ratio of similar ligands: {ratio} (TC >= {tc_threshold})')
