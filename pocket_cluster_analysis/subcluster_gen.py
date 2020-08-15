import numpy as np


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-main_cluster_file',
                        default='../../data/googlenet-classes',
                        required=False,
                        help='text file to get the cluster labels')

    parser.add_argument('-similarity_mat_dir',
                        default='../../similarity_matrices/',
                        required=False,
                        help='directory of output similarity matrix.')

    parser.add_argument('-subclusters_dir',
                        default='../../further_divided_clusters/',
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
    similarity_mat_dir = args.similarity_mat_dir
    subclusters_dir = args.subclusters_dir

    clusters_to_divide = [0] # list of clusters to be divided into subclusters

    similarity_mat = np.load(similarity_mat_dir)
    print(similarity_mat)

    ap_clustering = AffinityPropagation().fit(similarity_mat)
    subcluster_labels = ap_clustering.labels_
    print(subcluster_labels)

    divided_subclusters = gen_subclusters(cluster, subcluster_labels)

    del all_clusters[0]

    for subcluster in divided_subclusters:
        all_clusters.insert(0, subcluster) # insert the sub cluster at beginning.

    # save the new clusters
    with open('./new_clusters_by_dividing_cluster_0.json', 'w') as fp:
        json.dump(all_clusters, fp)




    '''
    # pocket set has all the pockets that
    similarity_mat_dict, pocket_set = read_similarity_file(similarity_file)


    similarity_mat = similarity_mat_gen(clusters[0], similarity_mat_dict)

    clusters, similarity_mat = remove_nan(clusters[0], similarity_mat)
    '''