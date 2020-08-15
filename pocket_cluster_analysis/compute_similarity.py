"""
Divide the large clusters into subclusters.
"""
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations 
from sklearn.cluster import AffinityPropagation
import json


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-main_cluster_file',
                        default='../../data/googlenet-classes',
                        required=False,
                        help='text file to get the cluster labels')

    parser.add_argument('-similarity_file',
                        default='../../data/googlenet-ssc.dat',
                        required=False,
                        help='file contains similarities between pockets')

    parser.add_argument('-cluster_number',
                        default=0,
                        required=False,
                        help='which cluster used to compute similarity matrix')

    parser.add_argument('-out_similarity_mat_dir',
                        default='../../similarity_matrices/similarity_matrix_cluster_0.npy',
                        required=False,
                        help='directory of output similarity matrix.')


    return parser.parse_args()


def read_cluster_file(cluster_file_dir):
    """
    Read the clustered pockets as a list of lists.
    """
    f = open(cluster_file_dir, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')
    
    while('' in data_text):
        data_text.remove('')

    clusters = []
    #cluster_sizes = []
    for x in data_text:
        cluster = x.split()[3:]
        cluster = [x.split(',')[0] for x in cluster]
        clusters.append(cluster)

    return clusters


def read_similarity(cluster, similarity_file):
    # read the similarities
    f = open(similarity_file, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')    
    while('' in data_text): 
        data_text.remove('')

    # the similarity matrix
    num_pockets = len(cluster)
    similarity_mat = np.zeros((num_pockets, num_pockets))
    #similarity_mat[:] = np.nan
    for idx in range(num_pockets):
        similarity_mat[idx][idx] = 1 # set diagonal to 1
    
    # put the each similarity into the matrix
    print('reading similarities from file to matrix...')
    for x in tqdm(data_text):
        x = x.split()
        pocket_a = x[0]
        pocket_b = x[1]
        similarity = float(x[2])

        # if the pair is in this cluster
        if (pocket_a in cluster) and (pocket_b in cluster):
            idx_a = cluster.index(pocket_a) # pocket_a's location
            idx_b = cluster.index(pocket_b) # pocket_b's location
            similarity_mat[idx_a, idx_b] = similarity
            similarity_mat[idx_b, idx_a] = similarity

    return similarity_mat


def remove_nan(cluster, similarity_mat):
    """
    Find the posisitons of nans and remove them from the similarity matrix. The corresponding
    pockets in the cluster are also removed.
    """
    # delete rows and columns with nans
    print('shape of similarity matrix before including nans: ', similarity_mat.shape)
    nan_idx = np.nonzero(np.isnan(similarity_mat))
    nan_rows = list(set(nan_idx[0]))
    nan_cols = list(set(nan_idx[1]))
    assert(nan_rows == nan_cols)    
    print('number of rows with nans: ', len(nan_rows))
    similarity_mat = np.delete(similarity_mat, nan_rows, 0)
    similarity_mat = np.delete(similarity_mat, nan_cols, 1)
    print('shape of similarity matrix after including nans: ', similarity_mat.shape)

    # delete corresponding pockets
    pockets_to_delete = []
    for pocket_idx in nan_rows:
        pockets_to_delete.append(cluster[pocket_idx])
    
    for pocket in pockets_to_delete:
        cluster.remove(pocket)      
    print('remaining number of pockets: ', len(cluster))

    return cluster, similarity_mat


if __name__=="__main__":
    args = get_args()
    main_cluster_file = args.main_cluster_file
    similarity_file = args.similarity_file
    cluster_number = args.cluster_number
    out_similarity_mat_dir = args.out_similarity_mat_dir

    all_clusters = read_cluster_file(main_cluster_file)
    #cluster_length = [len(x) for x in clusters]
    
    similarity_mat = read_similarity(all_clusters[cluster_number], similarity_file)
    np.save(out_similarity_mat_dir, similarity_mat)



