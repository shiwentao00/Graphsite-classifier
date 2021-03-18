"""
Label the unseen data, and save pocket names in a list of list. 
The order of the classes is exactly the same as the old data. 
"""
import yaml
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
import pandas as pd


if __name__ == "__main__":
    
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  

    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    merge_info = config['merge_info']

    # pocket clusters
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)   

    # convert seen data into list of sets
    cluster_sets = [set(x) for x in clusters]

    # list of empty lists to hold new data
    unseen_data = [[] for _ in range(num_classes)]

    # load the unseen-seen pairs
    df = pd.read_csv('../unseen-data/unseen_data_TC_1')
    print(df)
    unseen = df['pdb_unseen'].tolist()
    seen = df['pdb_seen'].tolist()
    unseen_dict = dict(zip(unseen, seen))
    #unseen_keys = list(unseen_dict.keys())

    for key in unseen_dict.keys():
        for cnt, cluster in enumerate(cluster_sets):
            if unseen_dict[key] in cluster:
                unseen_data[cnt].append(key)

    with open('../unseen-data/unseen-pocket-list.yaml', 'w') as file:
        yaml.dump(unseen_data, file)
                