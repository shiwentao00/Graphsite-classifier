"""
Dock each pocket with the 14 chosen ligands. The predicted class is the class
with highest docking score.
"""
import yaml
from os import listdir
from os.path import isfile, join


# pdbqt files of 14 label lignads for each class
label_to_ligand = {
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


def parse_search_config(path):
    """Parse the earch configuation files.

       Returns (x, y, z centers, x, y, z dimensions)
    """
    f = open(path, 'r')
    info = f.readlines()
    f.close()
    centers = info[1]
    dimensions = info[2]
    centers = centers.split()
    dimensions = dimensions.split()
    out = []
    out.extend(centers[-3:])
    out.extend(dimensions[-3:])
    out = [float(x) for x in out]
    return out


if __name__ == "__main__":
    # get 14 label ligands
    label_ligands_dir = '../../vina/data/ligands/'
    label_ligands_paths = []
    for i in range(14):
        label_ligands_paths.append(
            label_ligands_dir + label_to_ligand[i] + '.pdbqt')

    # 14 classes of pockets
    with open('../train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cluster_file_dir = '../../data/clusters_after_remove_files_with_no_popsa.yaml'
    pocket_dir = config['pocket_dir']
    merge_info = config['merge_info']
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # only 21124 search space configuration files
    search_config_dir = '../../vina/data/COG/'
    search_config_files = [f for f in listdir(
        search_config_dir) if isfile(join(search_config_dir, f))]
    print(search_config_files)
    print('number of search space configuration files: ', len(search_config_files))

    # target labels of all data points
    target = []
    # prediction by pocket matching
    prediction = []
    # for each class
    for label, cluster in enumerate(clusters):
        print('computing class {}...'.format(label))
        # for each pocket
        for pocket in cluster:
            # there are several missing search configuration files
            if pocket + '.out' in search_config_files:
                scores = []  # docking scores for each class
                # find its center and dimensions

                # defer the protein path

                # for each label ligands
                for ligand in range(14):
                    pass
                    # compute docking score (the lower the better)

                    # compute predicted class

                    # append results

                    # compute classification report
