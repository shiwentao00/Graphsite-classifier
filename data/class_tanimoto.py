"""Compute the stats of intra-Tanimoto coefficients of each class"""
import yaml
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import FingerprintSimilarity
import pickle


def compute_similarity(fp_a, fp_b):
    """compute the similarity between two smiles fingerprints"""
    return FingerprintSimilarity(fp_a, fp_b)


if __name__ == "__main__":
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
    for k, cluster in enumerate(clusters):
        print(f'computing fingerprints for cluster {k}...')
        total, success = 0, 0
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
    tanimoto_total = []
    for k, cluster in enumerate(clusters):
        print(f'computing cluster {k}...')
        tanimoto = []
        total, success = 0, 0
        for i, pocket in enumerate(tqdm(cluster)):
            for j in range(i + 1, len(cluster)):
                total += 1
                another_pocket = cluster[j]
                if pocket in fps and another_pocket in fps:
                    fp_a = fps[pocket]
                    fp_b = fps[another_pocket]
                    if fp_a and fp_b:
                        tanimoto.append(compute_similarity(fp_a, fp_b))
                        success += 1

        tanimoto_total.append(tanimoto)

        success_rate = success / total
        print(f'success rate: {success_rate}')

    with open('./data/tanimoto.pkl', 'wb') as f:
        pickle.dump(tanimoto_total, f)
