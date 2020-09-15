"""
Generate embeddings from a trained model for train/val/test pockets.
"""
import random
import argparse
import os
import torch
import numpy as np
from dataloader import read_cluster_file_from_yaml, select_classes, divide_clusters, pocket_loader_gen, cluster_by_chem_react
from dataloader import merge_clusters
from model import SiameseNet
from model import ResidualSiameseNet
from model import SelectiveSiameseNet
import yaml


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-embedding_dir',
                        default='../embeddings/',
                        required=False,
                        help='text file to get the cluster labels')                        

    parser.add_argument('-cluster_file_dir',
                        default='../data/clusters_after_remove_files_with_no_popsa.yaml',
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

    parser.add_argument('-run',
                        required=True,
                        type=int,
                        help='which experiment.')     

    parser.add_argument('-subcluster_file',
                        default='./pocket_cluster_analysis/results/subclusters_0.yaml',
                        required=False,
                        help='subclusters by chemical reaction of some clusters')                 

    return parser.parse_args()


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


if __name__=="__main__":
    random.seed(666) # deterministic sampled pockets and pairs from dataset
    print('seed: ', 666)    
    args = get_args()
    embedding_root = args.embedding_dir
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir
    run = args.run

    trained_model_dir = '../trained_models/trained_model_{}.pt'.format(run)
    print('using trained model: {}'.format(trained_model_dir))

    subcluster_file = args.subcluster_file
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)  

    #merge_info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #merge_info = [[0, 9], 1, 2, [3, 8], 4, 5, 6, 7]
    #merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7]
    #merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7, 10, 11, 12, 13]
    merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, 10]
    #merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, 10, 14, 15, 16, 17, 18]
    print('how to merge clusters: ', merge_info)

    subclustering = False # whether to further subcluster data according to subcluster_dict
    print('whether to further subcluster data according to chemical reaction: {}'.format(subclustering))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    batch_size = 4
    print('batch size:', batch_size)
    
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    num_classes = 14
    print('number of classes:', num_classes)
    cluster_th = 10000 # threshold of number of pockets in a class

    # normalize embeddings or not
    normalize = True

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)    

    # replace some clusters with their subclusters
    if subclustering == True:
        clusters, cluster_ids = cluster_by_chem_react(clusters, subcluster_dict)
        num_classes = len(clusters)
        print('number of classes after further clustering: ', num_classes)
    else:
        cluster_ids = [str(x) for x in merge_info] # use original cluster ids

    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    # missing popsa files for sasa feature at this moment
    #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'r', 'theta', 'phi', 'sequence_entropy'] 
    #features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'sequence_entropy'] 
    #features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'sasa', 'sequence_entropy']
    features_to_use = ['r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
                       'binding_probability', 'sequence_entropy']
    #features_to_use = ['r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
    #                   'binding_probability', 'sequence_entropy']

    # load trained model
    model = ResidualSiameseNet(num_features=len(features_to_use), dim=48, train_eps=True, num_edge_attr=1).to(device)
    #model = SiameseNet(num_features=len(features_to_use), dim=48, train_eps=True, num_edge_attr=1).to(device)
    #model = SelectiveSiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)

    model.load_state_dict(torch.load(trained_model_dir))
    model.eval()

    for which_split in ['train', 'val', 'test']:
        print('generating embeddings for {}...'.format(which_split))
        embedding_dir = embedding_root + 'run_{}/'.format(run)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        embedding_name = which_split + '_embedding' + '.npy'
        label_name = which_split + '_label' + '.npy'
        embedding_path = embedding_dir + embedding_name
        label_path = embedding_dir + label_name
        print('embedding path: ', embedding_path)
        print('label path: ', label_path)

        if which_split == 'train':
            clusters_to_embed = train_clusters
        elif which_split == 'val':
            clusters_to_embed = val_clusters
        elif which_split == 'test':
            clusters_to_embed = test_clusters

        # train loader, used to compute the geometric center of the embeddings of each cluster
        dataloader, dataloader_length = pocket_loader_gen(pocket_dir=pocket_dir, 
                                         pop_dir=pop_dir,
                                         clusters=clusters_to_embed, 
                                         features_to_use=features_to_use, 
                                         batch_size=batch_size, 
                                         shuffle=False, 
                                         num_workers=num_workers)
        
        embeddings, labels, cluster_set = compute_embeddings(dataloader, model, device, normalize=True)

        print('shape of generated embeddings: {}'.format(embeddings.shape))
        print('shape of labels: {}'.format(labels.shape))
        np.save(embedding_path, embeddings)
        np.save(label_path, labels)
