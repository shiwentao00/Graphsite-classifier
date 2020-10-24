"""
Load the pre-trained model, inference on the test set, print out the misclassified info
of a specified cluster.
"""
import yaml
import random
import torch
from dataloader import read_cluster_file_from_yaml, divide_clusters
from dataloader import PocketDataset, read_pocket
from dataloader import merge_clusters
from torch_geometric.data import Data, DataLoader
from model import MoNet
from tqdm import tqdm

class PocketDatasetwithName(PocketDataset):
    """dataset class with each data containing pocket name."""
    def __init__(self, pocket_dir, pop_dir, clusters, features_to_use):
        super(PocketDatasetwithName, self).__init__(pocket_dir=pocket_dir, 
                                                    pop_dir=pop_dir, 
                                                    clusters=clusters, 
                                                    features_to_use=features_to_use)

    def __getitem__(self, idx):
        pocket = self.pockets[idx]
        label = self.class_labels[idx]
        pocket_dir = self.pocket_dir + pocket + '/' + pocket + '.mol2'
        profile_dir = self.pocket_dir + pocket + '/' + pocket[0:-2] + '.profile'
        pop_dir = self.pop_dir + pocket[0:-2] + '.pops'

        x, edge_index, edge_attr = read_pocket(pocket_dir, profile_dir, pop_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)
        data =  Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
        data.name = pocket
        return data


if __name__ == "__main__":
    # which experiment
    run = 23

    # hardware to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    #print('device: ', device)

    # recreate dataset with the same split as when training
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  

    seed = config['seed']
    random.seed(seed) 

    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']

    merge_info = config['merge_info']
    features_to_use = config['features_to_use']
    num_features = len(features_to_use)

    batch_size = 4
    num_workers = 4

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    #print('number of pockets in training set: ', num_train_pockets)
    #print('number of pockets in validation set: ', num_val_pockets)
    #print('number of pockets in test set: ', num_test_pockets)
    #print('first 5 pockets in train set of cluster 0 before merging (to verify reproducibility):')
    #print(train_clusters[0][0:5])
    #print('first 5 pockets in val set of cluster 0 before merging (to verify reproducibility):')
    #print(val_clusters[0][0:5])
    #print('first 5 pockets in test set of cluster 0 before merging (to verify reproducibility):')
    #print(test_clusters[0][0:5])

    testset = PocketDatasetwithName(pocket_dir=pocket_dir, pop_dir=pop_dir, clusters=test_clusters, features_to_use=features_to_use)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # load the model
    which_model = 'jk'
    model_size = 96
    num_layers = 5
    assert which_model in ['jk', 'residual', 'normal', 'pna']

    model = MoNet(num_classes=num_classes, num_features=num_features, dim=model_size, 
                  train_eps=True, num_edge_attr=1, which_model=which_model, num_layers=num_layers, deg=None).to(device)

    trained_model_dir = config['trained_model_dir'] + 'trained_classifier_model_{}.pt'.format(run)
    model.load_state_dict(torch.load(trained_model_dir, map_location=torch.device('cpu')))

    # inference
    model.eval()

    preds = [] # all the predictions for the epoch
    labels = [] # all the labels for the epoch
    names = []
    for data in tqdm(testloader):
        #print(data.name)

        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = output.max(dim=1)[1]
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics

        preds.extend(pred_cpu)
        labels.extend(label)
        names.extend(data.name)

    # output the resutls
    #print(preds)
    #print(labels)
    #print(names)
    
    morpholine_ring_as_atp = []
    morpholine_ring_as_carbonhydrate = []
    for name, label, pred in zip(names, labels, preds):
        if label == 12 and pred == 0:
            morpholine_ring_as_atp.append(name)
        elif label == 12 and pred == 2:
            morpholine_ring_as_carbonhydrate.append(name)
    
    print('morpholine ring misclassified as ATP:')
    for x in morpholine_ring_as_atp:
        print(x)

    print('morpholine ring misclassified as Carbonhydrate:')
    for x in morpholine_ring_as_carbonhydrate:
        print(x)
            