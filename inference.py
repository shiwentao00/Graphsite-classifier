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

    parser.add_argument('-trained_model_dir',
                        default='../trained_models/trained_model_6.pt',
                        required=False,
                        help='directory to store the trained model.')                        

    return parser.parse_args()

def compute_geo_center(train_loader, model):
    """Compute the geometric centers of clusters in the training datset.
       The centers will be used as anchor points for classification."""
    embeddings = []
    labels = []
    for data in train_loader:
        data = data.to(device)
        labels.append(data.y.cpu().detach().numpy())
        embedding = model.get_embedding(data=data, normalize=True)
        embeddings.append(embedding.cpu().detach().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    cluster_set = list(set(labels)) # list of the clusters/classes

    class_centers = {}
    for cluster in cluster_set:
        cluster_idx = np.nonzero(labels == cluster)[0] # indices of the embeddings that belong to this cluster
        cluster_embedding = embeddings[cluster_idx] # embeddings of this cluster
        cluster_embedding = np.mean(cluster_embedding, axis=0) # geometric center of the embeddings
        class_centers.update({cluster:cluster_embedding})
    return class_centers


if __name__=="__main__":
    random.seed(666) # seed has to be the same as seed in train.py to generate the same clusters
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    trained_model_dir = args.trained_model_dir

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

    # read the original clustered pockets
    clusters = read_cluster_file(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

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
                                    clusters=train_clusters, 
                                    features_to_use=features_to_use, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers)

    # load trained model
    model = SiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)
    model.load_state_dict(torch.load(trained_model_dir))
    model.eval()

    # compute geometric centers of classes in train set
    class_centers = compute_geo_center(train_loader, model)

    # test loader
    '''
    test_loader = pocket_loader_gen(pocket_dir=pocket_dir, 
                                  clusters=test_clusters, 
                                  features_to_use=features_to_use, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers)
    '''

