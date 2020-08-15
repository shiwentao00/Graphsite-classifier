import numpy as np

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
    similarity_mat = np.load('./similarity_matrix_cluster_0_with_nan.npy')
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