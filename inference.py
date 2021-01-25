"""
Load a trained model and inference on unseen data.
"""
import os
import torch
import yaml
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from dataloader import read_pocket
from model import DeepDruG
import sklearn.metrics as metrics


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
        profile_dir = self.pocket_dir + pocket[0:-2] + '.profile'
        pop_dir = self.pocket_dir + pocket[0:-2] + '.pops'

        x, edge_index, edge_attr = read_pocket(
            pocket_dir, profile_dir, pop_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, y=torch.tensor([label]))
        return data, pocket


if __name__ == "__main__":
    # set of further filtered unseen data
    unseen_data_dir = '../unseen-data/unseen_pdb/'
    pockets = []
    for f in os.listdir(unseen_data_dir):
        if f.endswith(".mol2"):
            pockets.append(f[0:7])
    pockets = set(pockets)

    # list of unseen data
    with open('../unseen-data/unseen-pocket-list_new.yaml') as f:
        clusters = yaml.load(f, Loader=yaml.FullLoader)
    filtered_clusters = []
    for cluster in clusters:
        filtered_clusters.append([])
        for x in cluster:
            if x in pockets:
                filtered_clusters[-1].append(x)

    # for i in range(3, 14):
    #    filtered_clusters[i] = []

    # dataloader for unseen data
    features_to_use = ['x', 'y', 'z', 'r', 'theta', 'phi', 'sasa',
                       'charge', 'hydrophobicity', 'binding_probability', 'sequence_entropy']

    pocket_loader, _, _ = pocket_loader_gen(pocket_dir=unseen_data_dir,
                                            clusters=filtered_clusters,
                                            features_to_use=features_to_use,
                                            batch_size=1, shuffle=False, num_workers=1)

    # load model in cpu mode
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')  # detect cpu or gpu
    #print('device: ', device)
    which_model = config['which_model']
    model_size = config['model_size']
    num_layers = config['num_layers']
    num_channels = config['num_channels']
    num_features = len(features_to_use)
    num_classes = len(clusters)
    model = DeepDruG(num_classes=num_classes, num_features=num_features, dim=model_size,
                     train_eps=True, num_edge_attr=1, which_model=which_model, num_layers=num_layers,
                     num_channels=num_channels, deg=None).to(device)
    model.load_state_dict(torch.load('../trained_models/trained_classifier_model_63.pt',
                                     map_location=torch.device('cpu')))

    model.eval()

    # inference
    targets = []
    predictions = []
    for data, pocket_name in pocket_loader:
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = output.max(dim=1)[1]

        pred_cpu = list(pred.cpu().detach().numpy())
        label = list(data.y.cpu().detach().numpy())

        predictions.extend(pred_cpu)
        targets.extend(label)
        print(pocket_name[0], label[0], pred_cpu[0])

    # classification report
    report = metrics.classification_report(targets, predictions, digits=4)
    print('---------------classification report----------------')
    print(report)

    """
                precision    recall  f1-score   support

           0     0.9286    0.6842    0.7879        19
           1     1.0000    1.0000    1.0000         9
           2     0.6250    0.8333    0.7143         6
           3     0.0000    0.0000    0.0000         0
           5     0.5000    0.3333    0.4000         3
           6     0.2000    0.5000    0.2857         2
           7     0.0000    0.0000    0.0000         1
           8     1.0000    1.0000    1.0000         1
           9     0.0000    0.0000    0.0000         0
          10     0.0000    0.0000    0.0000         2
          12     0.3333    0.5000    0.4000         2

    accuracy                         0.6889        45
   macro avg     0.4170    0.4410    0.4171        45
weighted avg     0.7547    0.6889    0.7073        45
    """
