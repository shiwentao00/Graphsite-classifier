"""
Classify pockets based on the distances to medoids.
"""
import random
import argparse
import os
import numpy as np
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics
import yaml


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-embedding_dir',
                        default='../embeddings/',
                        required=False,
                        help='text file to get the cluster labels')                        

    parser.add_argument('-run',
                        required=True,
                        type=int,
                        help='which experiment.')     

    return parser.parse_args()


def compute_geo_centers(embeddings, labels):
    """Compute the geometric medians of clusters (medois) in the training datset.
       The medoids will be used as anchor points for classification."""
    # list of the clusters/classes
    cluster_set = list(set(labels)) 
    cluster_set.sort()

    class_medoids = []
    for cluster in cluster_set:
        cluster_idx = np.nonzero(labels == cluster)[0] # indices of the embeddings that belong to this cluster
        cluster_embedding = embeddings[cluster_idx] # embeddings of this cluster
        #cluster_center = np.mean(cluster_embedding, axis=0) # geometric center of the embeddings
        cluster_center = compute_medoids(cluster_embedding)
        class_medoids.append(cluster_center)
    class_medoids = np.vstack(class_medoids)
    return class_medoids


def compute_medoids(embeddings):
    """Compute the geometric median of the points. It is expected to be 
    robust against outliers."""
    dist_mat = cdist(embeddings, embeddings, metric='euclidean') # pair-wise distances
    similarity_sum = np.sum(dist_mat, axis=0) # summing up 
    medoid = embeddings[np.argmin(similarity_sum)] # get the one that is closest to its neighbors
    return medoid


def compute_report(embeddings, labels, class_medoids):
    """Compute the evaluation metrics."""
    distances_to_centers = cdist(embeddings, class_medoids)

     # same label as closest center
    predictions = np.argmin(distances_to_centers, axis=1)

    #acc = metrics.accuracy_score(labels, predictions)
    report = metrics.classification_report(labels, predictions)

    return report 


def compute_top3_acc(embeddings, labels, class_medoids):
    """Compute the top-5 classification accuracy."""
    # list of the clusters/classes
    cluster_set = list(set(labels)) 
    cluster_set.sort()
    
    distances_to_centers = cdist(embeddings, class_medoids)

    predictions = np.argsort(distances_to_centers, axis=1) # same label as closest center
    predictions = predictions[:,0:3] # first 5 columns

    total_data = embeddings.shape[0]
    correct = 0 
    for idx, row in enumerate(predictions):
        if labels[idx] in list(row):
            correct = correct + 1
    acc = correct / total_data
    return acc 


if __name__=="__main__":
    args = get_args()
    embedding_root = args.embedding_dir
    run = args.run
    embedding_dir = embedding_root + 'run_{}/'.format(run)
    
    merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7]
    print('how to merge clusters: ', merge_info)

    for which_split in ['train', 'val', 'test']:
        print('computing evaluation metrics for {}'.format(which_split))
        embedding_name = which_split + '_embedding' + '.npy'
        label_name = which_split + '_label' + '.npy'
        embedding_path = embedding_dir + embedding_name
        label_path = embedding_dir + label_name
        print('embedding path: ', embedding_path)
        print('label path: ', label_path)

        embeddings = np.load(embedding_path)
        labels = np.load(label_path)
        labels = labels.astype(int)
        
        # compute geometric centers of classes in train set
        if which_split=='train':
            class_medoids = compute_geo_centers(embeddings, labels)

        # train accuracy
        report = compute_report(embeddings, labels, class_medoids)
        print('{} report: ', which_split)
        print(report)

        top3_acc = compute_top3_acc(embeddings, labels, class_medoids)
        print('top-3 {} acc: {}'.format(which_split, top3_acc))
        print('----------------------------------------------------------')



    

