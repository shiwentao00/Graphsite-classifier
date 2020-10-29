"""
Train the siamese-architectured model in two phases. Paris are sampled
Uniformly from the combinations of classes in phase 1. In phase 2, only 
the hardest pairs are selected fro training
"""
import random
import yaml
import json
import os
import torch
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
from dataloader import divide_clusters_train_test, gen_pairs
from dataloader import dataloader_gen, pocket_loader_gen
from model import SiameseNet, ContrastiveLoss

import copy
from model import SelectiveSiameseNet, SelectiveContrastiveLoss
from dataloader import sample_from_list

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import numpy as np


def compute_embeddings(dataloader, model, device, normalize=True):
    """
    Compute embeddings, labels, and set of labels/clusters of a dataloader.
    """
    embeddings = []
    labels = []
    for cnt, data in enumerate(dataloader):
        data = data.to(device)
        labels.append(data.y.cpu().detach().numpy())
        embedding = model.get_embedding(data=data, normalize=normalize)
        embeddings.append(embedding.cpu().detach().numpy())
        #if cnt == 200:
        #    break
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    cluster_set = list(set(labels)) # list of the clusters/classes
    return embeddings, labels, cluster_set


def train_by_random_pairs():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_pair_loader, train_pair_size, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    # learning rate decay
    if epoch == pair_lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    total_loss = 0
    for data in train_pair_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding_a, embedding_b = model(data)

        loss = loss_function(embedding_a, embedding_b, data.y)
        loss.backward()
        
        # last incomplete batch is dropped, so just use batch_size
        total_loss += loss.item() * pair_batch_size 
        
        optimizer.step()
    train_loss = total_loss / train_pair_size
    return train_loss


def train_by_hard_pairs():
    """
    Train the model for 1 epoch, then return the mean loss of the data 
    in this epoch.
    Global vars: sampled_train_loader, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    # learning rate decay
    if epoch == selective_lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    # begin to select hard pairs for training
    if epoch == select_hard_pair_epoch:
        loss_function.set_select_hard_pairs(True)

    total_loss = 0
    num_loss_elements = 0 # total number of pairs used for training in this epoch
    for data in sampled_train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding = model(data)
        loss_mean, loss_sum, loss_shape = loss_function(embedding, data.y)
        loss_mean.backward()
        optimizer.step()

        # the loss is averaged over samples in a mini-batch
        # last incomplete batch is dropped, so just use batch_size
        total_loss += loss_sum.item()
        num_loss_elements += loss_shape[0]
    train_loss = total_loss / num_loss_elements

    return train_loss


def validate_by_knn_acc():
    """
    Validate the training performance by k-nearest neighbor 
    accuracy on the validation set.
    """
    model.eval()

    # embeddings of train pockets
    train_embedding, train_label, _ = compute_embeddings(train_loader, model, device, normalize=True)

    # embeddings of validation pockets
    val_embedding, val_label, _ = compute_embeddings(test_loader, model, device, normalize=True)

    # knn model
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(train_embedding, train_label)
    train_prediction = knn.predict(train_embedding)
    val_prediction = knn.predict(val_embedding)
    train_acc = metrics.accuracy_score(train_label, train_prediction)
    val_acc = metrics.accuracy_score(val_label, val_prediction)

    return train_acc, val_acc


def test_by_knn():
    """
    Run the trained model on the test split. 
    """
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(train_embedding, train_label)

    train_prediction = knn.predict(train_embedding)
    test_prediction = knn.predict(test_embedding)
    train_acc = metrics.accuracy_score(train_label, train_prediction)
    test_acc = metrics.accuracy_score(test_label, test_prediction)
    print('train accuracy: {}, validation accuracy: {}, test accuracy: {}'.format(train_acc, val_acc, test_acc))

    train_report = metrics.classification_report(train_label, train_prediction, digits=4)
    test_report = metrics.classification_report(test_label, test_prediction, digits=4)

    print('train report:')
    print(train_report)
    print('test report: ')
    print(test_report)
    

if __name__ == "__main__":
    with open('./two_phase_train.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    seed = config['seed']
    random.seed(seed)
    print('seed: ', seed)
    
    run = config['run']
    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']
    trained_model_dir = config['trained_model_dir'] + 'trained_model_{}.pt'.format(run)
    loss_dir = config['loss_dir'] + 'train_results_{}.json'.format(run)
    print('save trained model at: ', trained_model_dir)
    print('save loss at: ', loss_dir)

    merge_info = config['merge_info']
    train_pos_th = config['train_pos_th']
    train_neg_th = config['train_neg_th']
    cluster_sample_th = config['cluster_sample_th']
    features_to_use = config['features_to_use']
    print('how to merge clusters: ', merge_info)
    print('positive training pair sampling threshold: ', train_pos_th)
    print('negative training pair sampling threshold: ', train_neg_th)
    print('features to use: ', features_to_use)

    pair_num_epoch = config['pair_num_epoch']
    pair_lr_decay_epoch = config['pair_lr_decay_epoch'] 
    pair_batch_size = config['pair_batch_size']
    pair_learning_rate = config['pair_learning_rate']
    pair_weight_decay = config['pair_weight_decay']
    normalize = config['normalize'] # whether to normalize the embeddings in constrastive loss
    print('number of epochs to train:', pair_num_epoch)
    print('learning rate decay to half at epoch {}.'.format(pair_lr_decay_epoch))

    # margins for the relaxed contrastive loss
    similar_margin = 0.0
    dissimilar_margin = 2.0

    num_workers = os.cpu_count()
    num_workers = int(min(pair_batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    # read all the original clusters
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)    

    # divide the clusters into train, validation and test
    train_clusters, test_clusters = divide_clusters_train_test(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in test set: ', num_test_pockets)
    print('first 5 pockets in train set of cluster 0 before merging (to verify reproducibility):')
    print(train_clusters[0][0:5])
    print('first 5 pockets in test set of cluster 0 before merging (to verify reproducibility):')
    print(test_clusters[0][0:5])

    # uniformly sampled pairs for training
    train_pos_pairs, train_neg_pairs = gen_pairs(clusters=train_clusters, pos_pair_th=train_pos_th, neg_pair_th=train_neg_th)
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    train_pair_size = len(train_pos_pairs) +  len(train_neg_pairs)

    # train dataloader for phase 1 
    train_pair_loader = dataloader_gen(pocket_dir, 
                                       pop_dir,
                                       train_pos_pairs, 
                                       train_neg_pairs, 
                                       features_to_use, 
                                       pair_batch_size, 
                                       shuffle=True,
                                       num_workers=num_workers)

    # configurations for selecting hard pairs for training
    selective_num_epoch = config['selective_num_epoch']
    selective_lr_decay_epoch = config['selective_lr_decay_epoch']
    select_hard_pair_epoch = config['select_hard_pair_epoch']
    selective_batch_size = config['selective_batch_size']
    num_hard_pos_pairs = config['num_hard_pos_pairs'] # number of hardest similar pairs sampled from a mini-batch
    num_hard_neg_pairs = config['num_hard_neg_pairs'] # number of hardest dissimilar pairs sampled from a mini-batch
    selective_learning_rate = config['selective_learning_rate']
    selective_weight_decay = config['selective_weight_decay']
    print('number of epochs to train for hard pairs: ', selective_num_epoch)
    print('learning rate decay at epoch for hard pairs: ', selective_lr_decay_epoch)
    print('begin to select hard pairs at epoch {}'.format(select_hard_pair_epoch))
    print('batch size for hard pairs: ', selective_batch_size)
    print('number of hardest positive pairs for each mini-batch: ', num_hard_pos_pairs)
    print('number of hardest negative pairs for each mini-batch: ', num_hard_neg_pairs)

    # train dataloader for validation of phase 1
    train_loader, _, _ = pocket_loader_gen(pocket_dir=pocket_dir,
                                             pop_dir=pop_dir,
                                             clusters=train_clusters,
                                             features_to_use=features_to_use,
                                             batch_size=selective_batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    # test dataloader for knn for phase 1 & 2
    test_loader, _, _ = pocket_loader_gen(pocket_dir=pocket_dir,
                                               pop_dir=pop_dir,
                                               clusters=test_clusters,
                                               features_to_use=features_to_use,
                                               batch_size=num_workers,
                                               shuffle=False,
                                               num_workers=num_workers)

    print('\n*******************************************************')
    print('             train by random pairs')
    print('*******************************************************')
    which_model = config['which_model']
    model_size = config['model_size']
    num_layers = config['num_layers']
    assert which_model in ['jk', 'residual', 'normal']
    model = SiameseNet(num_features=len(features_to_use), dim=model_size, 
                           train_eps=True, num_edge_attr=1, which_model=which_model, num_layers=num_layers).to(device)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=pair_learning_rate, weight_decay=pair_weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    # differentiable, no parameters to train.
    loss_function = ContrastiveLoss(similar_margin=similar_margin, dissimilar_margin=dissimilar_margin, normalize=normalize, mean=True).to(device)  
    print('loss function:')
    print(loss_function)

    # training histories of both phases
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0

    # begin training
    for epoch in range(1, pair_num_epoch+1):
        train_loss = train_by_random_pairs()
        train_losses.append(train_loss)
        
        train_acc, val_acc = validate_by_knn_acc()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print('epoch: {}, train loss: {}, train acc: {}, validation acc: {}.'.format(epoch, train_loss, train_acc, val_acc))
        
        #if epoch > lr_decay_epoch: # store results for epochs after decay learning rate
        if  val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model.embedding_net.state_dict()) # save the weights of embedding_net 
            torch.save(model.state_dict(), trained_model_dir)

    print('best validation acc {} at epoch {}.\n'.format(best_val_acc, best_val_epoch))

    print('\n*******************************************************')
    print('             train by hard pairs')
    print('*******************************************************')
    # initialize the selective model and load weights to its EmbeddingNet
    model = SelectiveSiameseNet(num_features=len(features_to_use),
                                dim=model_size, train_eps=True, 
                                num_edge_attr=1, which_model=which_model, num_layers=num_layers).to(device)
    model.embedding_net.load_state_dict(best_model)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(
    model.parameters(), lr=selective_learning_rate, weight_decay=selective_weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    # differentiable, no parameters to train.
    loss_function = SelectiveContrastiveLoss(
        similar_margin=similar_margin, dissimilar_margin=dissimilar_margin, 
        num_pos_pair=num_hard_pos_pairs, num_neg_pair=num_hard_neg_pairs).to(device)
    print('loss function:')
    print(loss_function)

    # begin training, append results to previous lists
    for epoch in range(1, selective_num_epoch+1):
        # sample each class evenly 
        sampled_train_clusters = []
        for cluster in train_clusters:
            sampled_train_clusters.append(sample_from_list(cluster, cluster_sample_th))

        # re-generate train-loader
        sampled_train_loader, _, _ = pocket_loader_gen(pocket_dir=pocket_dir,
                                                    pop_dir=pop_dir,
                                                    clusters=sampled_train_clusters,
                                                    features_to_use=features_to_use,
                                                    batch_size=selective_batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

        # train
        train_loss = train_by_hard_pairs()
        train_losses.append(train_loss)
        
        # validate
        train_acc, val_acc = validate_by_knn_acc()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print('epoch: {}, train loss: {}, train acc: {}, validation acc: {}.'.format(epoch, train_loss, train_acc, val_acc))
        
        if  val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model.embedding_net.state_dict()) # save the weights of embedding_net 
            torch.save(model.state_dict(), trained_model_dir)

    print('best validation acc {} at epoch {}.'.format(best_val_acc, best_val_epoch))

    # write loss history to disk
    results = {'train_losses': train_losses, 'train_accs': train_accs, 'val_accs': val_accs}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

    print('\n*******************************************************')
    print('             k-nearest neighbor for testing')
    print('*******************************************************')
    model.embedding_net.load_state_dict(best_model) # load the best model with highest val acc
    model.eval()

    # embeddings of train pockets
    train_embedding, train_label, _ = compute_embeddings(train_loader, model, device, normalize=True)
    
    # embeddings of test pockets
    test_embedding, test_label, _ = compute_embeddings(test_loader, model, device, normalize=True)

    # knn testing
    test_by_knn()

    # save embeddings and labels
    embedding_dir = config['embedding_dir'] + 'run_{}/'.format(run)
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    for which_split in ['train', 'test']:
        print('generating embeddings for {}...'.format(which_split))
        embedding_name = which_split + '_embedding' + '.npy'
        label_name = which_split + '_label' + '.npy'
        embedding_path = embedding_dir + embedding_name
        label_path = embedding_dir + label_name
        print('embedding path: ', embedding_path)
        print('label path: ', label_path)

        if which_split == 'train':
            embedding = train_embedding 
            label = train_label
        elif which_split == 'test':
            embedding = test_embedding 
            label = test_label

        print('shape of generated embedding: {}'.format(embedding.shape))
        print('shape of label: {}'.format(label.shape))
        np.save(embedding_path, embedding)
        np.save(label_path, label)

    print('\nprogram finished.')