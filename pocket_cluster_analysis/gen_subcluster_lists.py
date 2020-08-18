"""
Generate the subclustering results of clusters in a yaml file.
"""
import argparse
import numpy as np
from sklearn_extra.cluster import KMedoids
from subclustering import similarity_to_distance, read_cluster_file
import yaml


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-main_cluster_file',
                        default='../../data/googlenet-classes',
                        required=False,
                        help='text file to get the cluster_id labels')

    parser.add_argument('-output_file',
                        default='./results/subclusters_0_9.yaml',
                        required=False,
                        help='path to the yaml file containing subcluster information')

    parser.add_argument('-similarity_mat_dir',
                        default='../../similarity_matrices/',
                        required=False,
                        help='directory of output similarity matrix.')

    return parser.parse_args()


def gen_subclusters(cluster, labels):
    """
    Gnerate subclusters as list of lists according to labels.
    """
    subclusters = []
    label_set = list(set(labels))
    for label in label_set:
        label_idx = np.nonzero(labels == label)[0]
        subcluster = []
        for idx in label_idx:
            subcluster.append(cluster[idx])
        subclusters.append(subcluster)
    return subclusters


if __name__=="__main__":
    args = get_args()
    main_cluster_file = args.main_cluster_file
    output_file = args.output_file
    similarity_mat_dir = args.similarity_mat_dir

    # all the original clusters by ligands
    all_clusters = read_cluster_file(main_cluster_file)

    # best configurations for kmedoids clustering
    '''
    clusters = [0, 1, 5, 7, 8, 9, 10, 11, 16, 24]
    configs = {0:[4, 96], # [k, seed] for kmedoids
               1:[2, 116],
               5:[3, 14],
               7:[2, 30],
               8:[4, 70],
               9:[4, 263],
               10:[4, 47],
               11:[4, 399],
               16:[5, 285],
               24:[5, 68]}
    '''
    clusters = [0, 1, 5, 7, 8, 9]
    configs = {0:[4, 96], # [k, seed] for kmedoids
               1:[2, 116],
               5:[3, 14],
               7:[2, 30],
               8:[4, 70],
               9:[4, 263]}

    # run kmedois at best configuration
    subcluster_dict = {} # all subcluster info, keys are cluster_ids
    for cluster_id in clusters:
        print('re-clustering cluster {}...'.format(cluster_id))

        # generate similarity matrix and distance matrix
        similarity_mat_path = similarity_mat_dir + 'similarity_matrix_cluster_' + str(cluster_id) + '.npy'
        similarity_mat = np.load(similarity_mat_path)
        dist_mat = similarity_to_distance(similarity_mat)

        # kmedoids at best configuraton
        config = configs[cluster_id]
        k = config[0]
        seed = config[1]
        print('k: {}, seed: {}'.format(k, seed))
        kmedoids_clustering = KMedoids(n_clusters=k, init='k-medoids++', metric='precomputed', random_state=seed).fit(dist_mat)
        subcluster_labels = kmedoids_clustering.labels_
        
        # generate subclusters according to labels
        subclusters = gen_subclusters(all_clusters[cluster_id], subcluster_labels)

        # put subclusters into the dictionary
        subcluster_dict.update({cluster_id: subclusters})

    print('saving subclusters to {}...'.format(output_file))
    with open(output_file, 'w') as file:
        documents = yaml.dump(subcluster_dict, file)