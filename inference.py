"""
Classify unseean pockets with trained similary model. This module contains:
    1. Dataset class for classification.
    2. Function that computes geometric center of the embeddings of each class.
    3. Module that computes the classification accuracy.   
"""
import random
import argparse
from dataloader import read_cluster_file, select_classes, divide_clusters


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
                        default='../trained_models/trained_model.pt/',
                        required=False,
                        help='directory to store the trained model.')                        

    return parser.parse_args()


if __name__=="__main__":
    random.seed(666) # seed has to be the same as seed in train.py to generate the same clusters
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    trained_model_dir = args.trained_model_dir

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

    print(train_clusters[-1])

    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    '''
