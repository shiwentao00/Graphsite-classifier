"""Find similar and disimilar pairs within a certain class"""
import argparse
import yaml
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import FingerprintSimilarity
import os


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-cls',
                        required=True,
                        type=int,
                        help='which class to find pairs.')

    parser.add_argument('-out',
                        required=False,
                        default='./data/class_tanimoto_pairs/',
                        help='random seed for splitting dataset.')

    return parser.parse_args()


def compute_similarity(fp_a, fp_b):
    """compute the similarity between two smiles fingerprints"""
    return FingerprintSimilarity(fp_a, fp_b)


if __name__ == "__main__":
    args = get_args()
    cls = int(args.cls)
    out = args.out

    # load the 14 classes
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cluster_file_dir = config['cluster_file_dir']
    merge_info = config['merge_info']
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # load the pocket-smile dictionary
    with open('./data/pocket-smiles.yaml') as f:
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

    # compute pair-wise tanimoto coefficients
    similar_pairs, different_pairs = [], []
    total, success = 0, 0
    print('computing tanimoto similarities...')
    for i, pocket in enumerate(tqdm(cluster)):
        for j in range(i + 1, len(cluster)):
            total += 1
            another_pocket = cluster[j]
            if pocket in fps and another_pocket in fps:
                fp_a = fps[pocket]
                fp_b = fps[another_pocket]
                if fp_a and fp_b:
                    s = compute_similarity(fp_a, fp_b)
                    if s >= 0.9:
                        similar_pairs.append((pocket, another_pocket))
                    elif s <= 0.35:
                        different_pairs.append((pocket, another_pocket))
                    success += 1
    success_rate = success / total
    print(f'success rate: {success_rate}')

    # store results to file
    similar_file = f'class_{cls}_similar_pairs.out'
    similar_file = os.path.join(out, similar_file)
    with open(similar_file, 'w') as f:
        for pair in similar_pairs:
            f.write(pair[0] + ' ' + pair[1] + '\n')

    different_file = f'class_{cls}_different_pairs.out'
    different_file = os.path.join(out, different_file)
    with open(different_file, 'w') as f:
        for pair in different_pairs:
            f.write(pair[0] + ' ' + pair[1] + '\n')
