# Copyright: Wentao Shi, 2020
import random
import itertools
import torch
from graphsite import PocketToGraph
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
import yaml

pocket_to_graph = PocketToGraph()

def read_pocket(mol_path, profile_path, pop_path,
                hydrophobicity, binding_probability,
                features_to_use, threshold):
    """Read the pocket as a Pytorch-geometric graph."""
    node_feature, edge_index, edge_attr = pocket_to_graph(
        mol_path=mol_path, 
        profile_path=profile_path, 
        pop_path=pop_path,
        hydrophobicity=hydrophobicity, 
        binding_probability=binding_probability,
        features_to_use=features_to_use, 
        threshold=threshold
    )

    # convert the data to pytorch tensors
    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return node_feature, edge_index, edge_attr


def dataloader_gen(pocket_dir, pop_dir, pos_pairs, neg_pairs, features_to_use,
                   batch_size, shuffle=True, num_workers=1):
    """Dataloader used to wrap PairDataset. Used for training and validation """
    dataset = PairDataset(
        pocket_dir=pocket_dir, pop_dir=pop_dir,
        pos_pairs=pos_pairs, neg_pairs=neg_pairs,
        features_to_use=features_to_use
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, follow_batch=['x_a', 'x_b'],
        drop_last=True
    )

    return dataloader


def pocket_loader_gen(pocket_dir, pop_dir, clusters,
                      features_to_use, batch_size,
                      shuffle=True, num_workers=1):
    """Dataloader used to wrap PocketDataset."""
    pocketset = PocketDataset(
        pocket_dir=pocket_dir,
        pop_dir=pop_dir,
        clusters=clusters,
        features_to_use=features_to_use
    )

    pocketloader = DataLoader(
        pocketset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return pocketloader, len(pocketset), pocketset


class PocketDataset(Dataset):
    """Dataset to generate single pocket graphs for inference/testing."""

    def __init__(self, pocket_dir, pop_dir, clusters, features_to_use):
        self.pocket_dir = pocket_dir
        self.pop_dir = pop_dir
        self.clusters = clusters

        # distance threshold to form an undirected edge between two atoms
        self.threshold = 4.5

        # hard coded hydrophobicity node feature
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
            'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
            'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
            'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
            'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }

        # hard coded binding probability node feature
        self.binding_probability = {
            'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
            'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
            'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
            'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
            'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884
        }

        total_features = [
            'x', 'y', 'z',
            'r', 'theta', 'phi',
            'sasa', 'charge',
            'hydrophobicity',
            'binding_probability',
            'sequence_entropy'
        ]

        # features to use should be subset of total_features
        assert(set(features_to_use).issubset(set(total_features)))
        self.features_to_use = features_to_use

        self.class_labels = []
        self.pockets = []
        for label, cluster in enumerate(self.clusters):
            # flatten the clusters list
            self.pockets.extend(cluster)
            for pocket in cluster:
                # class labels for all the pockets
                self.class_labels.append(label)

    def __len__(self):
        cluster_lengths = [len(x) for x in self.clusters]
        return sum(cluster_lengths)

    def __getitem__(self, idx):
        pocket = self.pockets[idx]
        label = self.class_labels[idx]
        pocket_dir = self.pocket_dir + pocket + '/' + pocket + '.mol2'
        profile_dir = self.pocket_dir + pocket + \
            '/' + pocket[0:-2] + '.profile'
        pop_dir = self.pop_dir + pocket[0:-2] + '.pops'

        x, edge_index, edge_attr = read_pocket(
            pocket_dir, profile_dir,
            pop_dir,
            self.hydrophobicity,
            self.binding_probability,
            self.features_to_use,
            self.threshold
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label])
        )

        return data


class PairDataset(Dataset):
    """Dataset to generate pairs of data for training and validation."""

    def __init__(self, pocket_dir, pop_dir, pos_pairs, neg_pairs, features_to_use):
        self.pocket_dir = pocket_dir
        self.pop_dir = pop_dir
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs
        self.num_pos_pairs = len(pos_pairs)
        self.num_neg_pairs = len(neg_pairs)
        self.threshold = 4.5  # distance threshold to form an undirected edge between two atoms

        # hydrophobicities are hardcoded
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
            'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
            'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
            'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
            'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }

        # binding probabilities are hardcoded
        self.binding_probability = {
            'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
            'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
            'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
            'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
            'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884
        }

        total_features = [
            'x', 'y', 'z',
            'charge', 'hydrophobicity',
            'binding_probability',
            'r', 'theta', 'phi',
            'sasa', 'sequence_entropy'
        ]

        # features to use should be subset of total_features
        assert(set(features_to_use).issubset(set(total_features)))
        self.features_to_use = features_to_use

    def __len__(self):
        return len(self.pos_pairs) + len(self.neg_pairs)

    def __getitem__(self, idx):
        # get the pair
        assert(idx >= 0 and idx < (self.num_pos_pairs + self.num_neg_pairs))
        if idx >= self.num_pos_pairs:
            idx = idx - self.num_pos_pairs
            pair = self.neg_pairs[idx]

            # pair label
            y = torch.tensor([0])
        else:
            pair = self.pos_pairs[idx]

            # pair label
            y = torch.tensor([1], dtype=torch.long)

        # pocket a location
        pocket_a_dir = self.pocket_dir + pair[0] + '/' + pair[0] + '.mol2'
        profile_a_dir = self.pocket_dir + \
            pair[0] + '/' + pair[0][0:-2] + '.profile'
        pop_a_dir = self.pop_dir + pair[0][0:-2] + '.pops'

        # pocket b location
        pocket_b_dir = self.pocket_dir + pair[1] + '/' + pair[1] + '.mol2'
        profile_b_dir = self.pocket_dir + \
            pair[1] + '/' + pair[1][0:-2] + '.profile'
        pop_b_dir = self.pop_dir + pair[1][0:-2] + '.pops'

        x_a, edge_index_a, edge_attr_a = read_pocket(
            pocket_a_dir, profile_a_dir,
            pop_a_dir, self.hydrophobicity,
            self.binding_probability, self.features_to_use,
            self.threshold
        )

        x_b, edge_index_b, edge_attr_b = read_pocket(
            pocket_b_dir, profile_b_dir,
            pop_b_dir, self.hydrophobicity,
            self.binding_probability,
            self.features_to_use, self.threshold
        )

        data = PairData(
            x_a=x_a, edge_index_a=edge_index_a, edge_attr_a=edge_attr_a,
            x_b=x_b, edge_index_b=edge_index_b, edge_attr_b=edge_attr_b, y=y
        )

        return data


class PairData(Data):
    """Paired data type. Each object has 2 graphs."""

    def __init__(self, x_a, edge_index_a, edge_attr_a,
                 x_b, edge_index_b, edge_attr_b, y):
        super(PairData, self).__init__(y=y)
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


def divide_and_gen_pairs(cluster_file_dir, subcluster_dict,
                         num_classes, cluster_th,
                         train_pos_th, train_neg_th,
                         val_pos_th, val_neg_th):
    """
    Divide the dataset and generate pairs of pockets for train, 
    validation, and test.

    Arguments:
        cluster_file_dir: directory of the cluster file.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

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

    # train pairs
    train_pos_pairs, train_neg_pairs = gen_pairs(
        clusters=train_clusters,
        pos_pair_th=train_pos_th,
        neg_pair_th=train_neg_th
    )

    # validation pairs
    val_pos_pairs, val_neg_pairs = gen_pairs(
        clusters=val_clusters,
        pos_pair_th=val_pos_th,
        neg_pair_th=val_neg_th
    )

    return train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs


def read_cluster_file(cluster_file_dir):
    """Read the clustered pockets as a list of lists."""
    f = open(cluster_file_dir, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')

    while('' in data_text):
        data_text.remove('')

    clusters = []
    for x in data_text:
        cluster = x.split()[3:]
        cluster = [x.split(',')[0] for x in cluster]
        clusters.append(cluster)

    return clusters


def read_cluster_file_from_yaml(cluster_file_dir):
    """Read the clustered pockets as a list of lists."""
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)
    return clusters


def select_classes(clusters, num_classes, th):
    """
    Keep the relatively large clusters and limit the number 
    of pockets in super-large clusters.

    Arguments:
        clusters: list of pocket lists. The lists 
                  are already ranked according to length.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    selected_classes = []
    for i in range(num_classes):
        selected_classes.append(sample_from_list(clusters[i], th))

    return selected_classes


def sample_from_list(input_list, th):
    """
    Randomly samples th pockets if number of pockets larger than th. 
    """
    if len(input_list) > th:
        return random.sample(input_list, th)
    else:
        return input_list


def divide_clusters(clusters):
    """
    Shuffle and divide the clusters into train, validation and test
    """
    # shuffle the pockets in each cluster
    # random.shuffle happens inplace
    [random.shuffle(x) for x in clusters]  

    # sizes of the clusters
    cluster_sizes = [len(x) for x in clusters]
    # print('total pockets:{}'.format(sum(cluster_sizes)))

    # train
    train_sizes = [int(0.7 * x) for x in cluster_sizes]
    # print('train pockets:{}'.format(sum(train_sizes)))

    # validation
    val_sizes = [int(0.15 * x) for x in cluster_sizes]
    # print('val pockets:{}'.format(sum(val_sizes)))

    # test
    train_val_sizes = [sum(x) for x in zip(train_sizes, val_sizes)]
    test_sizes = [a - b for a, b in zip(cluster_sizes, train_val_sizes)]
    # print('test pockets:{}'.format(sum(test_sizes)))

    train_clusters = []
    val_clusters = []
    test_clusters = []
    for i in range(len(clusters)):
        train_clusters.append(clusters[i][0:train_sizes[i]])
        val_clusters.append(
            clusters[i][train_sizes[i]: train_sizes[i]+val_sizes[i]])
        test_clusters.append(clusters[i][train_sizes[i]+val_sizes[i]:])
    return train_clusters, val_clusters, test_clusters


def divide_clusters_train_test(clusters):
    """Shuffle and divide the clusters into train, validation and test"""
    # shuffle (inplace) the pockets in each cluster
    [random.shuffle(x) for x in clusters]

    # sizes of the clusters
    cluster_sizes = [len(x) for x in clusters]
    # print('total pockets:{}'.format(sum(cluster_sizes)))

    # train cluster sizes
    train_sizes = [int(0.8 * x) for x in cluster_sizes]

    train_clusters = []
    test_clusters = []
    for i in range(len(clusters)):
        train_clusters.append(clusters[i][0:train_sizes[i]])
        test_clusters.append(clusters[i][train_sizes[i]:])
    return train_clusters, test_clusters


def gen_pairs(clusters, pos_pair_th=1000, neg_pair_th=20):
    """Generate pairs of pockets from input clusters."""
    # generate pairs of pockets in the same cluster
    pos_pairs = []
    for cluster in clusters:
        pairs = list(itertools.combinations(cluster, 2))
        pairs = sample_from_list(pairs, pos_pair_th)
        pos_pairs.extend(pairs)

    # generate pairs of pockets in different clusters
    neg_pairs = []
    class_pairs = list(itertools.combinations(clusters, 2))
    for class_pair in class_pairs:
        pairs = [(x, y) for x in class_pair[0] for y in class_pair[1]]
        pairs = sample_from_list(pairs, neg_pair_th)
        neg_pairs.extend(pairs)

    return pos_pairs, neg_pairs


def cluster_by_chem_react(clusters, subcluster_dict):
    """
    Replace the original clusters with their subclusters.

    Arguments:

        clusters: list of lists of pockts.

        subcluster_dict: dictionary contains subclusters. Keys are the original cluster ids. 
    """
    num_original_clusters = len(clusters)
    dict_keys = list(subcluster_dict.keys())

    # store the unchanged clusters first
    new_cluster_dict = {}
    for original_cluster_id in range(num_original_clusters):
        if original_cluster_id not in dict_keys:  # if this cluster is not further divided
            new_cluster_dict.update(
                {'{}-0'.format(original_cluster_id): clusters[original_cluster_id]})

    # update the subclusters
    for key in dict_keys:
        subclusters = subcluster_dict[key]
        for idx, subcluster in enumerate(subclusters):
            new_cluster_dict.update({'{}-{}'.format(key, idx): subcluster})

    new_cluster_ids = list(new_cluster_dict.keys())
    new_cluster_ids.sort(key=lambda x: int(x.split('-')[0]))

    # for a list of lists of pockts according to new_cluster_ids
    new_clusters = []
    for cluster_id in new_cluster_ids:
        new_clusters.append(new_cluster_dict[cluster_id])

    return new_clusters, new_cluster_ids


def merge_clusters(clusters, merge_info):
    """Merge some clusters according to merge_info.

    Arguments:   
    clusters - list of lists of pockets to represent the pocket clusters.   

    merge_info - new combination of clusters. e.g., [[0,3], [1,2], 4].

    Return:   
    new_clusters -  list of lists of pockets to represent the pocket clusters after merging.
    """
    new_clusters = []
    for item in merge_info:
        if type(item) == int:
            new_clusters.append(clusters[item])
        elif type(item) == list:
            temp = []
            for idx in item:
                temp.extend(clusters[idx])
            new_clusters.append(temp)
        else:
            raise TypeError(
                "'merge_info' muse be a list with elements that are either int or list of int.")

    return new_clusters
