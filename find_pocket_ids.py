"""
Find all the pdb ids used in the project.
"""
import yaml
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters


if __name__ == "__main__":
    # recreate dataset with the same split as when training
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']    
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    for cluster in clusters:
        #print(len(cluster))
        for x in cluster:
            print(x) 