"""
Solve the problem with traditional end-to-end multi-class model.
"""
import argparse
import random
import os
import torch
import torch.nn as nn
from dataloader import read_cluster_file, select_classes, divide_clusters, pocket_loader_gen, cluster_by_chem_react
from model import MoNet
import numpy as np
import sklearn.metrics as metrics
import json
import yaml


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

    parser.add_argument('-pop_dir',
                        default='../data/pops-googlenet/',
                        required=False,
                        help='directory of popsa files for sasa feature')

    parser.add_argument('-subcluster_file',
                        default='./pocket_cluster_analysis/results/subclusters_0.yaml',
                        required=False,
                        help='subclusters by chemical reaction of some clusters')

    parser.add_argument('-trained_model_dir',
                        default='../trained_models/trained_model_classifier_1.pt/',
                        required=False,
                        help='directory to store the trained model.')                        

    parser.add_argument('-loss_dir',
                        default='./results/classifier_train_results_1.json/',
                        required=False,
                        help='directory to store the training losses.')

    return parser.parse_args()


def train():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()
    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(output, data.y)
        loss.backward()
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
        pred = output.max(dim=1)[1]

        # convert prediction and label to list
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics
        
        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    train_acc = metrics.accuracy_score(epoch_label, epoch_pred)# accuracy of entire epoch
    train_loss = loss_total / train_size # averaged training loss
    return train_loss, train_acc


def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    val_acc = metrics.accuracy_score(epoch_label, epoch_pred)# accuracy of entire epoch    
    val_loss = loss_total / val_size # averaged training loss
    return val_loss, val_acc


def test():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in test_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    test_acc = metrics.accuracy_score(epoch_label, epoch_pred)# accuracy of entire epoch    
    test_loss = loss_total / test_size # averaged training loss
    return test_loss, test_acc


def compute_class_weights(clusters):
    """Compute the weights of each class/cluster according to number of data.   

    clusters: list of lists of pockets."""
    cluster_lengths = [len(x) for x in clusters]
    cluster_weights = np.array([1/x for x in cluster_lengths])
    cluster_weights = cluster_weights/np.mean(cluster_weights) # normalize the weights with mean 
    #print(cluster_weights)
    return cluster_weights


if __name__=="__main__":
    random.seed(666) # deterministic sampled pockets and pairs from dataset
    print('seed: ', 666)
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir
    subcluster_file = args.subcluster_file
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)        
    
    trained_model_dir = args.trained_model_dir
    loss_dir = args.loss_dir
    
    num_classes = 10
    print('number of classes:', num_classes)
    cluster_th = 10000 # threshold of number of pockets in a class
    
    subclustering = False # whether to further subcluster data according to subcluster_dict
    print('whether to further subcluster data according to chemical reaction: {}'.format(subclustering))

    # tunable hyper-parameters
    num_epochs = 100
    print('number of epochs to train:', num_epochs)
    learning_rate = 0.0015
    weight_decay = 0.0005

    batch_size = 128
    print('batch size:', batch_size)
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    num_gpu = 0
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
    print('number of gpus: ', num_gpu)

    # read the original clustered pockets
    clusters = read_cluster_file(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # replace some clusters with their subclusters
    if subclustering == True:
        clusters, cluster_ids = cluster_by_chem_react(clusters, subcluster_dict)
        num_classes = len(clusters)
        print('number of classes after further clustering: ', num_classes)

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    # missing popsa files for sasa feature at this moment
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 
    num_features = len(features_to_use)

    train_loader, train_size = pocket_loader_gen(pocket_dir=pocket_dir, 
                                                 pop_dir=pop_dir,
                                                 clusters=train_clusters, 
                                                 features_to_use=features_to_use, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 num_workers=num_workers)

    val_loader, val_size = pocket_loader_gen(pocket_dir=pocket_dir, 
                                             pop_dir=pop_dir,
                                             clusters=val_clusters, 
                                             features_to_use=features_to_use, 
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers) 

    test_loader, test_size = pocket_loader_gen(pocket_dir=pocket_dir, 
                                             pop_dir=pop_dir,
                                             clusters=test_clusters, 
                                             features_to_use=features_to_use, 
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers) 

    model = MoNet(num_classes=num_classes, num_features=num_features, dim=32, train_eps=True, num_edge_attr=1).to(device)
    print('model architecture:')
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    class_weights = compute_class_weights(train_clusters)
    class_weights = torch.FloatTensor(class_weights).to(device)
    loss_function = nn.NLLLoss(weight=class_weights)
    print('loss function:')
    print(loss_function)
    
    best_val_loss = 9999999
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    print('begin training...')
    for epoch in range(1, 1+num_epochs):
        train_loss, train_acc = train()
        val_loss, val_acc = validate()
        test_loss, test_acc = test()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print('epoch: {}, train loss: {}, acc: {}, val loss: {}, acc: {}, test loss: {}, acc: {}'.format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

        if  val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), trained_model_dir)

    print('best val loss {} at epoch {}.'.format(best_val_loss, best_val_epoch))

    results = {'train_losses': train_losses, 'train_accs': train_accs, 'val_losses': val_losses, 'val_accs': val_accs}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

