"""
Dataloader for the Siamese graph neural network. 
"""
import random
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader


class PairData(Data):
    """
    Paired data type. Each object has 2 graphs.
    """
    def __init__(self,x_a, edge_index_a, edge_attr_a, x_b, edge_index_b, edge_attr_b):
        super(PairData, self).__init__()
        self.x_a = x_a
        self.edge_index_a = edge_index_a
        self.edge_attr_a = edge_attr_a

        self.x_b = x_b
        self.edge_index_b = edge_index_b        
        self.edge_attr_b = edge_attr_b

    def __inc__(self, key, value):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        else:
            return super(PairData, self).__inc__(key, value)
    
'''
class PairDataset(Dataset):
    def __init__(self, cluster_file_dir, num_classes, th):



    def __len__():
        return sum(self.cluster_sizes)

    def __gen_pos_pairs(self):

    def __gen_neg_pairs()
'''

def divide_and_gen_pairs(cluster_file_dir, num_classes, th):
    """
    Generate pairs of pockets for train, validation, and test.

    Arguments:
        cluster_file_dir: directory of the cluster file.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    # read the original clustered pockets
    clusters = read_cluster_file(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, th)

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)

    # train pairs

    # validation pairs

    # test pairs

    


def read_cluster_file(cluster_file_dir):
    """
    Read the clustered pockets as a list of lists.
    """
    f = open(cluster_file_dir, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')
    
    while('' in data_text): 
        data_text.remove('')

    clusters = []
    #cluster_sizes = []
    for x in data_text:
        cluster = x.split()[3:]
        cluster = [x.split(',')[0] for x in cluster]
        clusters.append(cluster)
        #cluster_sizes.append(len(cluster))

    return clusters


def select_classes(clusters, num_classes, th):
    """
    Keep the relatively large clusters and limit the number of pockets in super-large clusters.

    Arguments:
        clusters: list of pocket lists. The lists are already ranked according to length.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    selected_classes = []
    for i in range(num_classes):
        selected_classes.append(sample_from_list(clusters[i], th))

    #select_classe_lengths = [len(x) for x in selected_classes]

    return selected_classes
    

def sample_from_list(cluster, th):
    """
    Randomly samples th pockets if number of pockets larger than th. 
    """
    if len(cluster) > th:
        return random.sample(cluster, th)
    else:
        return cluster


def divide_clusters(clusters):
    """
    Shuffle and divide the clusters into train, validation and test
    """
    # shuffle the pockets in each cluster
    [random.shuffle(x) for x in clusters] # random.shuffle happens inplace

    # sizes of the clusters
    cluster_sizes = [len(x) for x in clusters]
    #print(cluster_sizes)

    # train
    train_sizes = [int(0.6 * x) for x in cluster_sizes]

    # validation
    val_sizes = [int(0.2 * x) for x in cluster_sizes]

    # test
    train_val_sizes = [sum(x) for x in zip(train_sizes, val_sizes)]
    test_sizes = [a - b for a, b in zip(cluster_sizes, train_val_sizes)]

    train_clusters = []
    val_clusters = []
    test_clusters = []
    for i in range(len(clusters)):
        train_clusters.append(clusters[i][0:train_sizes[i]])
        val_clusters.append(clusters[i][train_sizes[i]: train_sizes[i]+val_sizes[i]])
        test_clusters.append(clusters[i][train_sizes[i]+val_sizes[i]:])
    
    return train_clusters, val_clusters, test_clusters


if __name__=="__main__":
    cluster_file_dir = '../data/googlenet-classes'
    #dataset = PairDataset(cluster_file_dir=cluster_file_dir, num_classes=300, th=800)
    divide_and_gen_pairs(cluster_file_dir=cluster_file_dir, num_classes=150, th=800)



    #data = PairData(x_a, edge_index_a, edge_attr_a, x_b, edge_index_b, edge_attr_b)
    #data_list = [data, data]
    #loader = DataLoader(data_list, batch_size=2, follow_batch=['x_a', 'x_b'])
    #batch_data = next(iter(loader))
