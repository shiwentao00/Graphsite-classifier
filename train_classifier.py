"""
Solve the problem with traditional end-to-end multi-class model.
"""
import yaml
import random
import os
import torch
import torch.nn as nn
from dataloader import read_cluster_file_from_yaml, divide_clusters, pocket_loader_gen
from dataloader import merge_clusters
from model import MoNet, FocalLoss
import numpy as np
import sklearn.metrics as metrics
import json



def train():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()
    if epoch == lr_decay_epoch:
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

    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    

    run = config['run']
    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']
    trained_model_dir = config['trained_model_dir'] + 'trained_classifier_model_{}.pt'.format(run)
    loss_dir = config['loss_dir'] + 'train_classifier_results_{}.json'.format(run)    
    print('save trained model at: ', trained_model_dir)
    print('save loss at: ', loss_dir)

    merge_info = config['merge_info']
    features_to_use = config['features_to_use']
    num_features = len(features_to_use)
    print('how to merge clusters: ', merge_info)
    print('features to use: ', features_to_use)
    
    num_epoch = config['num_epoch']
    lr_decay_epoch = config['lr_decay_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    print('number of epochs: ', num_epoch)
    print('learning rate decay at epoch: ', lr_decay_epoch)
    print('batch size: ', batch_size)

    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)    

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    train_loader, train_size = pocket_loader_gen(pocket_dir=pocket_dir, 
                                                 pop_dir=pop_dir,
                                                 clusters=train_clusters, 
                                                 features_to_use=features_to_use, 
                                                 batch_size=batch_size, 
                                                 shuffle=True, 
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

    which_model = config['which_model']
    model_size = config['model_size']
    which_loss = config['which_loss']
    assert which_model in ['jk', 'residual', 'normal']
    assert which_loss in ['CrossEntropy', 'Focal']
    model = MoNet(num_classes=num_classes, num_features=num_features, dim=model_size, 
                  train_eps=True, num_edge_attr=1, which_model=which_model).to(device)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    if which_loss == 'CrossEntropy':
        class_weights = compute_class_weights(train_clusters)
        class_weights = torch.FloatTensor(class_weights).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif which_loss == 'Focal':
        loss_function = FocalLoss(gamma=2, reduction='mean')
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
    for epoch in range(1, 1 + num_epoch):
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

