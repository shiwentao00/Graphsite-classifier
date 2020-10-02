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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sklearn.metrics as metrics
import json
import copy
import matplotlib
import matplotlib.pyplot as plt


def train():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()

    # learning rate delay
    #if epoch in lr_decay_epoch:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.5 * param_group['lr']

    # increasing gamma of FocalLoss
    if which_loss == 'Focal' and focal_gamma_ascent == True:
        if epoch in focal_gamma_ascent_epoch:
            global gamma
            gamma += 1
            print('epoch {}, gamma increased to {}.'.format(epoch, gamma))
            loss_function.set_gamma(gamma)

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


def gen_classification_report(dataloader):
    """
    Generate a detailed classification report.
    """
    model.eval()

    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in dataloader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = output.max(dim=1)[1]
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    report = metrics.classification_report(epoch_label, epoch_pred, digits=4)
    confusion_mat = metrics.confusion_matrix(y_true=epoch_label, y_pred=epoch_pred, normalize='true') 
    return report, confusion_mat


if __name__=="__main__":
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  

    seed = config['seed']
    random.seed(666) 
    print('seed: ', 666)

    run = config['run']
    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']
    trained_model_dir = config['trained_model_dir'] + 'trained_classifier_model_{}.pt'.format(run)
    loss_dir = config['loss_dir'] + 'train_classifier_results_{}.json'.format(run)    
    confusion_matrix_dir = config['confusion_matrix_dir']
    print('save trained model at: ', trained_model_dir)
    print('save loss at: ', loss_dir)

    merge_info = config['merge_info']
    features_to_use = config['features_to_use']
    num_features = len(features_to_use)
    print('how to merge clusters: ', merge_info)
    print('features to use: ', features_to_use)
    
    num_epoch = config['num_epoch']
    #lr_decay_epoch = config['lr_decay_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    print('number of epochs: ', num_epoch)
    #print('learning rate decay at epoch: ', lr_decay_epoch)
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
    print('first 5 pockets in train set of cluster 0 before merging (to verify reproducibility):')
    print(train_clusters[0][0:5])
    print('first 5 pockets in val set of cluster 0 before merging (to verify reproducibility):')
    print(val_clusters[0][0:5])
    print('first 5 pockets in test set of cluster 0 before merging (to verify reproducibility):')
    print(test_clusters[0][0:5])

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
    num_layers = config['num_layers']
    which_loss = config['which_loss']
    assert which_model in ['jk', 'residual', 'normal']
    assert which_loss in ['CrossEntropy', 'Focal']
    model = MoNet(num_classes=num_classes, num_features=num_features, dim=model_size, 
                  train_eps=True, num_edge_attr=1, which_model=which_model, num_layers=num_layers).to(device)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    # decay learning rate when validation accuracy stops increasing.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, cooldown=40, min_lr=0.0001, verbose=True)
    print('learning rate scheduler: ')
    print(scheduler)

    if which_loss == 'CrossEntropy':
        class_weights = compute_class_weights(train_clusters)
        class_weights = torch.FloatTensor(class_weights).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif which_loss == 'Focal':
        gamma = config['initial_focal_gamma']
        print('initial gamma of FocalLoss: ', gamma)
        loss_function = FocalLoss(gamma=gamma, reduction='mean')
        focal_gamma_ascent = config['focal_gamma_ascent']
        if focal_gamma_ascent == True:
            focal_gamma_ascent_epoch = config['focal_gamma_ascent_epoch']
            print('increase gamma of FocalLoss at epochs: ', focal_gamma_ascent_epoch)
    print('loss function:')
    print(loss_function)
    
    best_val_acc = 0
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

        if  val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), trained_model_dir)

        scheduler.step(val_acc)

    print('best val acc {} at epoch {}.'.format(best_val_acc, best_val_epoch))

    # save the history of loss and accuracy
    results = {'train_losses': train_losses, 'train_accs': train_accs, 'val_losses': val_losses, 'val_accs': val_accs}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

    # load the best model to generate a detailed classification report
    print('****************************************************************')
    model.load_state_dict(best_model)
    train_report, train_confusion_mat = gen_classification_report(train_loader)
    val_report, val_confusion_mat = gen_classification_report(val_loader)
    test_report, test_confusion_mat = gen_classification_report(test_loader)
    
    font = {'size'   : 20}
    matplotlib.rc('font', **font)   

    print('train report:')
    print(train_report)
    print('train confusion matrix:')
    print(train_confusion_mat)
    fig, ax = plt.subplots(figsize=(28, 24))
    confusion_matrix_path = confusion_matrix_dir + 'confusion_matrix_{}_train.png'.format(run)
    metrics.ConfusionMatrixDisplay(train_confusion_mat, display_labels=None).plot(ax=ax)
    plt.savefig(confusion_matrix_path)
    print('---------------------------------------')
    
    print('validation report:')
    print(val_report)
    print('validation confusion matrix:')
    print(val_confusion_mat)
    fig, ax = plt.subplots(figsize=(28, 24))
    confusion_matrix_path = confusion_matrix_dir + 'confusion_matrix_{}_val.png'.format(run)
    metrics.ConfusionMatrixDisplay(val_confusion_mat, display_labels=None).plot(ax=ax)
    plt.savefig(confusion_matrix_path)
    print('---------------------------------------')
    
    print('test report: ')
    print(test_report)
    print('test confusion matrix:')
    print(test_confusion_mat)
    fig, ax = plt.subplots(figsize=(28, 24))
    confusion_matrix_path = confusion_matrix_dir + 'confusion_matrix_{}_test.png'.format(run)
    metrics.ConfusionMatrixDisplay(test_confusion_mat, display_labels=None).plot(ax=ax)
    plt.savefig(confusion_matrix_path)
    print('---------------------------------------')
    
    print('program finished.')

    


