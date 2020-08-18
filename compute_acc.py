"""
Classify unseean pockets with trained similary model. This module contains:
    1. Dataset class for classification.
    2. Function that computes geometric center of the embeddings of each class.
    3. Module that computes the classification accuracy.   
"""
import random
import argparse
import os
import torch
import numpy as np
from dataloader import read_cluster_file, select_classes, divide_clusters, pocket_loader_gen
from model import SiameseNet, ContrastiveLoss
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics
import yaml


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-cluster_file_dir',
                        default='../data/googlenet-classes',
                        required=False,
                        help='text file to get the cluster labels')

    parser.add_argument('-pocket_dir',
                        default='../data/googlenet-dataset/',
                        required=False,
                        help='directory of pockets')

    parser.add_argument('-pop_dir',
                        default='../data/pops-googlenet/',
                        required=False,
                        help='directory of popsa files for sasa feature')

    parser.add_argument('-subcluster_file',
                        default='./pocket_cluster_analysis/results/subclusters.yaml',
                        required=False,
                        help='subclusters by chemical reaction of some clusters')

    parser.add_argument('-trained_model_dir',
                        default='../trained_models/trained_model_7.pt',
                        required=False,
                        help='directory to store the trained model.')                        

    return parser.parse_args()


def compute_geo_centers(train_loader, model, device, normalize=True):
    """Compute the geometric centers of clusters in the training datset.
       The centers will be used as anchor points for classification."""
    embeddings, labels, cluster_set = compute_embeddings(train_loader, model, device, normalize)
    cluster_set.sort()# sort the labels

    class_centers = []
    for cluster in cluster_set:
        cluster_idx = np.nonzero(labels == cluster)[0] # indices of the embeddings that belong to this cluster
        cluster_embedding = embeddings[cluster_idx] # embeddings of this cluster
        #cluster_center = np.mean(cluster_embedding, axis=0) # geometric center of the embeddings
        cluster_center = compute_medoid(cluster_embedding)
        class_centers.append(cluster_center)
    class_centers = np.vstack(class_centers)
    return class_centers


def compute_medoid(embeddings):
    """Compute the geometric median of the points. It is expected to be 
    robust against outliers."""
    dist_mat = cdist(embeddings, embeddings, metric='euclidean') # pair-wise distances
    similarity_sum = np.sum(dist_mat, axis=0) # summing up 
    medoid = embeddings[np.argmin(similarity_sum)] # get the one that is closest to its neighbors
    return medoid


def compute_embeddings(dataloader, model, device, normalize=True):
    """
    Compute embeddings, labels, and set of labels/clusters of a dataloader.
    """
    embeddings = []
    labels = []
    for cnt, data in enumerate(dataloader):
        data = data.to(device)
        labels.append(data.y.cpu().detach().numpy())
        embedding = model.get_embedding(data=data, normalize=normalize)
        embeddings.append(embedding.cpu().detach().numpy())
        #if cnt == 200:
        #    break
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    cluster_set = list(set(labels)) # list of the clusters/classes
    return embeddings, labels, cluster_set


def compute_acc(dataloader, model, class_centers, device, normalize=True):
    """Compute the classification accuracy."""
    embeddings, labels, cluster_set = compute_embeddings(dataloader, model, device, normalize)
    
    distances_to_centers = cdist(embeddings, class_centers)

    predictions = np.argmin(distances_to_centers, axis=1) # same label as closest center

    acc = metrics.accuracy_score(labels, predictions)
    
    # compute the accuracy for each cluster
    for cluster in cluster_set:
        cluster_idx = np.nonzero(labels == cluster)[0]
        cluster_labels = labels[cluster_idx]
        cluster_predictions = predictions[cluster_idx]
        cluster_acc = metrics.accuracy_score(cluster_labels, cluster_predictions)
        print('cluster {} accuracy: {}.'.format(cluster, cluster_acc))
    return acc 


def compute_top5_acc(dataloader, model, class_centers, device, normalize=True):
    """Compute the top-5 classification accuracy."""
    embeddings, labels, cluster_set = compute_embeddings(dataloader, model, device, normalize)
    
    distances_to_centers = cdist(embeddings, class_centers)

    predictions = np.argsort(distances_to_centers, axis=1) # same label as closest center
    predictions = predictions[:,0:5] # first 5 columns

    total_data = embeddings.shape[0]
    correct = 0 
    for idx, row in enumerate(predictions):
        if labels[idx] in list(row):
            correct = correct + 1
    acc = correct / total_data
    return acc 


if __name__=="__main__":
    random.seed(666) # seed has to be the same as seed in train.py to generate the same clusters
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir

    subcluster_file = args.subcluster_file
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)    
    
    trained_model_dir = args.trained_model_dir
    print('computing classification accuracies of {}'.format(trained_model_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    batch_size = 4
    print('batch size:', batch_size)
    
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    num_classes = 60
    print('number of classes:', num_classes)
    cluster_th = 10000 # threshold of number of pockets in a class

    normalize = True

    # read the original clustered pockets
    clusters = read_cluster_file(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # replace some clusters with their subclusters
    clusters, cluster_ids = cluster_by_chem_react(clusters, subcluster_dict)

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    # missing popsa files for sasa feature at this moment
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 

    # train loader, used to compute the geometric center of the embeddings of each cluster
    train_loader = pocket_loader_gen(pocket_dir=pocket_dir, 
                                     pop_dir=pop_dir,
                                     clusters=train_clusters, 
                                     features_to_use=features_to_use, 
                                     batch_size=batch_size, 
                                     shuffle=False, 
                                     num_workers=num_workers)

    # load trained model
    model = SiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)
    model.load_state_dict(torch.load(trained_model_dir))
    model.eval()

    # compute geometric centers of classes in train set
    class_centers = compute_geo_centers(train_loader, model, device, normalize=normalize)

    # train accuracy
    train_acc = compute_acc(train_loader, model, class_centers, device, normalize=normalize)
    print('training accuracy: ', train_acc)
    top5_train_acc = compute_top5_acc(train_loader, model, class_centers, device, normalize=normalize)
    print('top-5 training accuracy: ', top5_train_acc)
    print('----------------------------------------------------------')

    # validation accuracy
    val_loader = pocket_loader_gen(pocket_dir=pocket_dir, 
                                   pop_dir=pop_dir,
                                   clusters=val_clusters, 
                                   features_to_use=features_to_use, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   num_workers=num_workers)    
    val_acc = compute_acc(val_loader, model, class_centers, device, normalize=normalize)
    print('validation accuracy: ', val_acc)
    top5_val_acc = compute_top5_acc(val_loader, model, class_centers, device, normalize=normalize)
    print('top-5 validation accuracy: ', top5_val_acc)
    print('----------------------------------------------------------')

    # test accuracy
    test_loader = pocket_loader_gen(pocket_dir=pocket_dir, 
                                   pop_dir=pop_dir,
                                   clusters=test_clusters, 
                                   features_to_use=features_to_use, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   num_workers=num_workers)    
    test_acc = compute_acc(test_loader, model, class_centers, device, normalize=normalize)
    print('test accuracy: ', test_acc)
    top5_test_acc = compute_top5_acc(test_loader, model, class_centers, device, normalize=normalize)
    print('top-5 test accuracy: ', top5_test_acc)
    print('----------------------------------------------------------')

    

