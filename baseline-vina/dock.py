"""
Dock each pocket with the 14 chosen ligands. The predicted class is the class
with highest docking score.
"""
import yaml
from os import listdir
from os.path import isfile, join
import subprocess
import sklearn.metrics as metrics


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


def dock(vina_path, protein_path, ligand_path, config, out_path):
    """
    Call Autodock VINA program to perform docking.
    Arguments:
        vina_path - path of autodock vina's executable
        protein_path - pdbqt file of protein
        ligand_path - pdbqt file of ligand
        config - tuple containing 6 numbers (x, y, z centers, x, y, z dimensions).

    Return:
        docking_score: free energy left, the lower the more robust the docking.
    """
    p = subprocess.run(vina_path + ' --receptor {} --ligand {}'.format(protein_path, ligand_path) +
                                   ' --center_x {} --center_y {} --center_z {}'.format(config[0], config[1], config[2]) +
                                   ' --size_x {} --size_y {} --size_z {}'.format(
                                       config[3], config[4], config[5]) +
                                   ' --out {}'.format(),
                       shell=True,
                       stdout=subprocess.PIPE,
                       text=True)
    # check=True,
    # cwd='/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/')  # working directory

    def parse_result(result):
        return float(result.split()[-1])

    # when there is no error
    if p.returncode == 0:
        result = p.stdout
        print(result)
        return parse_result(result)
    else:
        global error_cnt
        error_cnt += 1
        return 0


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


if __name__ == "__main__":
    # path of autodock vina's executable
    vina_path = '../../vina/software/autodock_vina_1_1_2_linux_x86/bin/vina'

    # number of errors occured during docking
    global error_cnt
    error_cnt = 0

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
    print('number of search space configuration files: ', len(search_config_files))

    # directory of proteins for docking
    protein_dir = '/home/wentao/Desktop/local-workspace/siamese-monet-project/vina/data/proteins/'

    # target labels and prediction of all data points
    target, prediction = [], []
    # for each class
    for label, cluster in enumerate(clusters):
        print('computing class {}...'.format(label))
        # for each pocket
        for pocket in cluster:
            # there are several missing search configuration files
            if pocket + '.out' in search_config_files:
                # add true lable to prediction list
                prediction.append(label)

                # find its center and dimensions
                search_config = parse_search_config(
                    search_config_dir + pocket + '.out')

                # defer the protein path
                protein_path = protein_dir + pocket[0:-2] + '.pdbqt'
                print(protein_path)
                scores = []  # docking scores for each class
                # for each label ligands
                for ligand in range(14):
                    # current ligand path
                    ligand_path = label_ligands_paths[ligand]

                    # compute docking score (the lower the better)
                    score = dock(vina_path, protein_path,
                                 ligand_path, search_config)

                    # compute predicted class

                    # append results

                    # compute classification report
                    break
            break
        break

    print('total number of errors during docking: ', error_cnt)
