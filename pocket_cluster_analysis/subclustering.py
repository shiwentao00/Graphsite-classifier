import argparse
import numpy as np
from compute_similarity import read_cluster_file
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


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

    parser.add_argument('-original_cluster',
                        type=int,
                        default=0,
                        required=False,
                        help='which original cluster to further divide into subclusters.')                        

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


def similarity_to_distance(similarity_mat):
    """convert similarity matrix to distance matrix"""
    return 1 - similarity_mat
    #return 1 / similarity_mat


def plot_silhouette_scores(original_cluster, silhouette_scores, cluster_labels, num_clusters):
    # sorted set of cluster numbers according to frequency in descending order
    cluster_set = sort_labels_by_frequency(cluster_labels)
    n_clusters = len(cluster_set)

    font = {'size'   : 16}
    matplotlib.rc('font', **font) 
    fig, ax = plt.subplots(figsize=(16, 16))
    y_lower = 10
    for cnt, cluster in enumerate(cluster_set): # plot 10 largest clusters only
        cluster_silhouette_scores = silhouette_scores[cluster_labels == cluster]
        cluster_silhouette_scores.sort()

        cluster_size = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + cluster_size        

        color = cm.nipy_spectral(float(cluster) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_scores,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples      

        if cnt >= num_clusters-1:
            break

    ax.set_title("Silhouette plot for the {} clusters.".format(num_clusters))
    ax.set_xlabel("Silhouette coefficient values")
    #ax.set_ylabel("Cluster label")  
    plt.savefig('./results/cluster_{}/cluster_{}_k{}_kmdoids_silhouette.png'.format(original_cluster, original_cluster, num_clusters))


def sort_labels_by_frequency(labels):
    """Sort the list of labels according to its frequency. 
    This funtion returns a list of label set sorted in ascent order frequency. """
    label_set = list(set(labels))
    labels = np.array(labels)

    # get the frequencies
    frequencies = {}
    for label in label_set:
        freq = np.count_nonzero(labels == label)
        frequencies.update({label:freq})

    # sort by frequency
    label_set.sort(key = lambda x: frequencies[x], reverse=True)
    return label_set


if __name__=="__main__":
    args = get_args()
    main_cluster_file = args.main_cluster_file
    similarity_mat_dir = args.similarity_mat_dir
    original_cluster = args.original_cluster
    subclusters_dir = args.subclusters_dir

    all_clusters = read_cluster_file(main_cluster_file)

    print('re-clustering cluster {}...'.format(original_cluster))
    similarity_mat_path = similarity_mat_dir + 'similarity_matrix_cluster_' + str(original_cluster) + '.npy'
        
    # generate similarity matrix and distance matrix
    similarity_mat = np.load(similarity_mat_path)
    dist_mat = similarity_to_distance(similarity_mat)

    for k in [2, 3, 4, 5, 6, 7, 8]: # number of clusters
        print('clustering for k={}'.format(k))
        best_score = -1 # worst score possible, so guaranteed to be better for first trial
        for seed in tqdm(range(500)): # 200 trials
            kmedoids_clustering = KMedoids(n_clusters=k, init='k-medoids++', metric='precomputed', random_state=seed).fit(dist_mat)
            subcluster_labels = kmedoids_clustering.labels_

            # evaluate clustering results
            silhouette_scores = metrics.silhouette_samples(dist_mat, subcluster_labels, metric='precomputed')
            avg_silhouette_score = np.mean(silhouette_scores)
            if avg_silhouette_score > best_score:
                best_score = avg_silhouette_score
                best_config = {'k':k, 'seed':seed, 'silhouette':avg_silhouette_score}
                best_clustering = kmedoids_clustering

        print('best configuration for k={}'.format(k))
        print(best_config)

        subcluster_labels = best_clustering.labels_
        silhouette_scores = metrics.silhouette_samples(dist_mat, subcluster_labels, metric='precomputed')

        plot_silhouette_scores(original_cluster, silhouette_scores, subcluster_labels, num_clusters=k)

    '''
    subclusters = gen_subclusters(all_clusters[cluster], subcluster_labels)
    subcluster_lengths = [len(x) for x in subclusters]
    print('length of the subclusters of cluster {}: '.format(cluster))
    subcluster_lengths.sort(reverse=True)
    print(subcluster_lengths)
    '''



    '''
    divided_subclusters = gen_subclusters(cluster, subcluster_labels)

    del all_clusters[0]

    for subcluster in divided_subclusters:
        all_clusters.insert(0, subcluster) # insert the sub cluster at beginning.

    # save the new clusters
    with open('./new_clusters_by_dividing_cluster_0.json', 'w') as fp:
        json.dump(all_clusters, fp)
    '''

    #db_clustering = DBSCAN(eps=0.4, metric='precomputed').fit(dist_mat)
    #subcluster_labels = db_clustering.labels_
    #k = len(list(set(subcluster_labels))) # get number of clusters

    #ap_clustering = AffinityPropagation(affinity='precomputed', verbose=True, random_state=666).fit(similarity_mat)
    #subcluster_labels = ap_clustering.labels_
    #k = len(list(set(subcluster_labels))) # get number of clusters

    #agg_clustering = AgglomerativeClustering(n_clusters=None ,affinity='precomputed', linkage='average', distance_threshold=0.8).fit(dist_mat)
    #subcluster_labels = agg_clustering.labels_
    #k = len(list(set(subcluster_labels))) # get number of clusters

