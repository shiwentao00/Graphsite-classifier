"""
Specify a random seed and cluster number, print all pockets in the testset.
"""
import random
import yaml
from dataloader import read_cluster_file_from_yaml, divide_clusters_train_test
from dataloader import merge_clusters

if __name__ == "__main__":
    seed = 23
    random.seed(seed) 

    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  

    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)
    train_clusters, test_clusters = divide_clusters_train_test(clusters)
    #print('first 5 pockets in train set of cluster 0 before merging (to verify reproducibility):')
    #print(train_clusters[0][0:5])
    #print('first 5 pockets in test set of cluster 0 before merging (to verify reproducibility):')
    #print(test_clusters[0][0:5])

    cluster = test_clusters[12]
    for pocket in cluster:
        print(pocket)