"""list the proteins used in this project"""
import yaml


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


if __name__ == "__main__":
    # recreate dataset with the same split as when training
    with open('../../train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cluster_file_dir = '../../../data/clusters_after_remove_files_with_no_popsa.yaml'
    pocket_dir = config['pocket_dir']
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # get the input files
    in_dir = '../../../data/googlenet-dataset/'
    protein_paths = []
    proteins = []
    for cluster in clusters:
        for x in cluster:
            proteins.append(x[0:5])
            protein_paths.append(
                '/googlenet-dataset/{}/{}.pdb'.format(x, x[0:5]))
    print('total number of pockets: ', len(proteins))

    # remove duplicates
    proteins = list(set(proteins))
    print('total number of proteins (after removing duplicates): ', len(proteins))

    # save proteins to a file
    with open('./protein_list.txt', 'w') as f:
        for protein in proteins:
            f.write(protein + '\n')
    
