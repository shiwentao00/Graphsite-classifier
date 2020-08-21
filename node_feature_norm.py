"""
Compute the mean and std of each feature of the entire dataset.
"""
import random
import os
import numpy as np
from dataloader import read_cluster_file, select_classes, pocket_loader_gen
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    random.seed(666) # deterministic sampled pockets and pairs from dataset
    cluster_file_dir = '../data/googlenet-classes'
    pocket_dir = '../data/googlenet-dataset/'
    pop_dir = '../data/pops-googlenet/'

    batch_size = 4    
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))

    num_classes = 60
    cluster_th = 10000 # threshold of number of pockets in a class

    clusters = read_cluster_file(cluster_file_dir)
    clusters = select_classes(clusters, num_classes, cluster_th)
    features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 
    #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 
    
    # all pockets are here
    data_loader, data_loader_size = pocket_loader_gen(pocket_dir=pocket_dir, 
                                     pop_dir=pop_dir,
                                     clusters=clusters, 
                                     features_to_use=features_to_use, 
                                     batch_size=batch_size, 
                                     shuffle=False, 
                                     num_workers=num_workers)

    # node array of entire dataset
    nodes = []
    for cnt, data in enumerate(data_loader):
        batch_nodes = data.x.detach().numpy()
        nodes.append(batch_nodes)
        #if cnt == 3:
        #    break
    nodes = np.vstack(nodes)
    print('nodes matrix shape: ')
    print(nodes.shape)
    # plot distributions of features
    num_features = nodes.shape[1]
    for ft in range(num_features):
        ft_vec = nodes[:,ft]
        print(features_to_use[ft])
        print(ft_vec)
        plt.figure()
        fig_dir = './feature_distributions/' + str(features_to_use[ft]) + '.png'
        sns.distplot(ft_vec, bins=60)
        plt.savefig(fig_dir)



    # compute mean and std for each feature/column

