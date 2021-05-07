"""
Dock selected pockets with the 14 label ligands
"""
import argparse
import pickle
import yaml
import os
import numpy as np
from dock import smina_dock

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-which_class',
                        type=int,
                        default=0,
                        required=True,
                        help='Which class of pockets to process, integer from 0 to 13.')

    parser.add_argument('-start',
                        type=int,
                        default=0,
                        required=True,
                        help='Start index of selected pockets in this class')

    parser.add_argument('-end',
                        type=int,
                        default=10,
                        required=True,
                        help='Start index of selected pockets in this class')
    return parser.parse_args()


def read_cluster_file_from_yaml(cluster_file_dir):
    """Read the clustered pockets as a list of lists.
    """
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)
    return clusters


def merge_clusters(clusters, merge_info):
    """Merge some clusters according to merge_info.
    Arguments:
    clusters - list of lists of pockets to represent the pocket clusters.
    merge_info - new combination of clusters. e.g., [[0,3], [1,2], 4].
    Return:
    new_clusters -  list of lists of pockets to represent the pocket clusters after merging.
    """
    new_clusters = []
    for item in merge_info:
        if type(item) == int:
            new_clusters.append(clusters[item])
        elif type(item) == list:
            temp = []
            for idx in item:
                temp.extend(clusters[idx])
            new_clusters.append(temp)
        else:
            raise TypeError(
                "'merge_info' muse be a list with elements that are either int or list of int.")

    return new_clusters


# pdbqt files of 14 label lignads for each class
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


if __name__ == "__main__":
    # which class of pockets to process
    args = get_args()
    which_class = args.which_class
    start = args.start
    end = args.end
    print('class: {}, start: {}, end: {}.'.format(which_class, start, end))

    # smina installed via conda
    smina_path = 'smina'

    # get 14 label ligands
    label_ligands_dir = '../../../smina/ligands/'
    label_ligands_paths = []
    for i in range(14):
        label_ligands_paths.append(
            label_ligands_dir + ligand_labels[i] + '.pdbqt'
        )

    # 14 classes of pockets
    with open('../../train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cluster_file_dir = '../../../data/clusters_after_remove_files_with_no_popsa.yaml'
    pocket_dir = config['pocket_dir']
    merge_info = config['merge_info']
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # directory of proteins for docking
    protein_dir = '../../../smina/proteins/'

    # directory to store output files
    out_dir = '../../../smina/output/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # pocket cneters
    pocket_center_path = '../pocket_center/pocket_center.pickle'
    with open(pocket_center_path, 'rb') as f:
        pocket_centers = pickle.load(f)

    # docking cubic sizes
    docking_box_path = '../docking_box/docking_boxes.yaml'
    with open(docking_box_path, 'r') as f:
        docking_boxes = yaml.full_load(f)

    # target labels and prediction of all data points
    target, prediction = [], []

    label = which_class
    cluster = clusters[label]
    print('computing class {}...'.format(label))

    if end > len(cluster):
        end = len(cluster)

    num_errors = 0
    for pocket in cluster[start:end + 1]:
        if pocket in pocket_centers:
            # add true lable to target list
            target.append(label)

            # path of protein
            protein = protein_dir + pocket[0:-2] + '.pdbqt'

            # center of docking search space
            pocket_center = pocket_centers[pocket]

            scores = []  # docking scores for each class
            # for each label ligands
            for ligand_class in range(14):
                # current ligand path
                ligand = label_ligands_paths[ligand_class]

                # docking box (cubic) size
                docking_box = docking_boxes[ligand_labels[ligand_class]]

                score, _ = smina_dock(
                    'smina', 
                    protein, 
                    ligand, 
                    pocket_center,
                    docking_box
                )

                scores.append(score)

            # this pocketed is excluded if something went wrong
            if None in scores:
                num_errors += 1
                continue

            # compute predicted class
            pred = np.argmin(np.array(scores))
            pred = pred.item()
            print(scores)
            # append results
            prediction.append(pred)

    # save the target and predicitons in a yaml file
    prediction_path = out_dir + 'preds-class'
    prediction_path += (str(label) + '-' + str(start) +
                        '-' + str(end) + '.yaml')
    # print(prediction)
    with open(prediction_path, 'w') as file:
        yaml.dump(prediction, file)

    # number of erroneous dockings
    print('total number of erroneous during docking: ', num_errors)


    
