"""
Generate embeddings and visualize them in 2-d plane
"""
import random
import argparse
import os
import torch
import numpy as np
from dataloader import read_cluster_file_from_yaml, select_classes, divide_clusters, pocket_loader_gen, cluster_by_chem_react
from dataloader import merge_clusters
from compute_acc import compute_embeddings
from model import SiameseNet
import yaml
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


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

    parser.add_argument('-trained_model_dir',
                        default='../trained_models/trained_model_27.pt',
                        required=False,
                        help='directory to store the trained model.')       

    parser.add_argument('-subcluster_file',
                        default='./pocket_cluster_analysis/results/subclusters_0.yaml',
                        required=False,
                        help='subclusters by chemical reaction of some clusters')                 

    return parser.parse_args()


def visualize_embeddings(embeddings, labels, cluster_ids, image_path):
    """Visualize 2d embeddings and color them by labels.
    """
    font = {'size'   : 16}
    matplotlib.rc('font', **font)   

    # list of the clusters/classes
    cluster_set = list(set(labels)) 
    cluster_set.sort()

    embedding_list = []
    label_list = []    
    for cluster in cluster_set:
        idx = np.nonzero(labels == cluster)[0]
        embedding_list.append(embeddings[idx,:])
        label_list.append(labels[idx])

    embedding_list = np.vstack(embedding_list)
    label_list = list(np.hstack(label_list))
    cluster_id_list = [cluster_ids[x] for x in label_list]

    fig = plt.figure(figsize=(12, 12))
    #colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink", "crimson", "bright orange"]
    #colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink", "crimson", "bright orange"]
    colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink"]
    #cust_palette = sns.color_palette("RdBu_r", len(list(set(cluster_id_list))))
    cust_palette = sns.xkcd_palette(colors)
    ax = sns.scatterplot(x=embedding_list[:,0], 
                         y=embedding_list[:,1], 
                         hue=cluster_id_list, 
                         markers='.', 
                         palette= cust_palette
                         )
    plt.savefig(image_path)


if __name__=="__main__":
    args = get_args()
    embedding_dir = args.embedding_dir
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir
    trained_model_dir = args.trained_model_dir
    print('using trained model: {}'.format(trained_model_dir))

    embed = False

    subcluster_file = args.subcluster_file
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)  

    # which split of dataset to embed. train/val/test
    which_split = 'val'
    print('which split: ', which_split)

    image_path = './{}_27.png'.format(which_split)
    print('image saved to: ', image_path)

    name = trained_model_dir.split('/')[-1]
    name = name.split('.')[0]
    name = name.split('_')[-1]
    embedding_name = which_split + name + '_embedding' + '.npy'
    label_name = which_split + name + '_label' + '.npy'
    embedding_path = embedding_dir + embedding_name
    label_path = embedding_dir + label_name
    print('embedding path: ', embedding_path)
    print('label path: ', label_path)

    #merge_info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    merge_info = [[0, 9], 1, 2, [3, 8], 4, 5, 6, 7]
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

    num_classes = 10
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

    # Generate embeddings of the given dataloader
    if embed == True:
        # divide the clusters into train, validation and test
        train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
        num_train_pockets = sum([len(x) for x in train_clusters])
        num_val_pockets = sum([len(x) for x in val_clusters])
        num_test_pockets = sum([len(x) for x in test_clusters])
        print('number of pockets in training set: ', num_train_pockets)
        print('number of pockets in validation set: ', num_val_pockets)
        print('number of pockets in test set: ', num_test_pockets)

        # missing popsa files for sasa feature at this moment
        #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 
        #features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'sequence_entropy'] 
        features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'sasa', 'sequence_entropy']
    
        # load trained model
        model = SiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)
        model.load_state_dict(torch.load(trained_model_dir))
        model.eval()

        # select which split to generate embedding
        assert(which_split in ['train', 'val', 'test'])
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
    else:
        # load the embeddings and labels
        embeddings = np.load(embedding_path)
        labels = np.load(label_path)
        labels = labels.astype(int)

        print('computing TSNE...')
        tsne_embedding = TSNE(n_components=2).fit_transform(embeddings)
        visualize_embeddings(tsne_embedding, labels, cluster_ids, image_path)