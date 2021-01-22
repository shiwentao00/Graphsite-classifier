"""
Load a trained model and inference on unseen data.
"""
import torch
import yaml
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from dataloader import read_pocket


def pocket_loader_gen(pocket_dir, clusters, features_to_use, batch_size, shuffle=False, num_workers=1):
    """Dataloader used to wrap PocketDataset."""
    pocketset = PocketDataset(pocket_dir=pocket_dir,
                              clusters=clusters,
                              features_to_use=features_to_use)

    pocketloader = DataLoader(pocketset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers,
                              drop_last=False)

    return pocketloader, len(pocketset), pocketset


class PocketDataset(Dataset):
    """Dataset to generate single pocket graphs for inference/testing."""

    def __init__(self, pocket_dir, clusters, features_to_use):
        self.pocket_dir = pocket_dir
        self.clusters = clusters
        self.threshold = 4.5  # distance threshold to form an undirected edge between two atoms

        # hard coded info to generate 2 node features
        self.hydrophobicity = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
                               'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
                               'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
                               'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
                               'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
        self.binding_probability = {'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
                                    'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
                                    'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
                                    'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
                                    'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884}

        total_features = ['x', 'y', 'z', 'r', 'theta', 'phi', 'sasa', 'charge',
                          'hydrophobicity', 'binding_probability', 'sequence_entropy']

        # features to use should be subset of total_features
        assert(set(features_to_use).issubset(set(total_features)))
        self.features_to_use = features_to_use

        self.class_labels = []
        self.pockets = []
        for label, cluster in enumerate(self.clusters):
            self.pockets.extend(cluster)  # flatten the clusters list
            for pocket in cluster:
                # class labels for all the pockets
                self.class_labels.append(label)

    def __len__(self):
        cluster_lengths = [len(x) for x in self.clusters]
        return sum(cluster_lengths)

    def __getitem__(self, idx):
        pocket = self.pockets[idx]
        label = self.class_labels[idx]
        pocket_dir = self.pocket_dir + pocket + '.mol2'
        profile_dir = self.pocket_dir + pocket + '.profile'
        pop_dir = self.pocket_dir + pocket[0:-2] + '.pops'

        x, edge_index, edge_attr = read_pocket(
            pocket_dir, profile_dir, pop_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, y=torch.tensor([label]))
        return data


if __name__ == "__main__":
    # list of unseen data
    with open('../unseen-data/unseen-pocket-list_new.yaml') as f:
        clusters = yaml.load(f, Loader=yaml.FullLoader)

    # dataloader for unseen data
    unseen_data_dir = '../unseen-data/unseen_pdb/'

    # load model in cpu mode

    # inference

    # compute metrics
