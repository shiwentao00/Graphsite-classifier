import argparse
import random
import os
import torch
from torch_geometric.nn import DataParallel
from dataloader import read_cluster_file_from_yaml, select_classes, divide_clusters, cluster_by_chem_react, gen_pairs
from dataloader import dataloader_gen
from dataloader import merge_clusters
from dataloader import pocket_loader_gen
from gen_embeddings import compute_embeddings
from model import SiameseNet, ContrastiveLoss
from model import ResidualSiameseNet
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import json
import yaml


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-cluster_file_dir',
                        default='../data/clusters_after_remove_files_with_no_popsa.yaml',
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
                        default='../trained_models/trained_model_.pt/',
                        required=False,
                        help='directory to store the trained model.')                        

    parser.add_argument('-loss_dir',
                        default='./results/train_results.json/',
                        required=False,
                        help='directory to store the training losses.')

    return parser.parse_args()


def train():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_pair_loader, train_pair_size, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    # learning rate decay
    if epoch == lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    total_loss = 0
    for data in train_pair_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding_a, embedding_b = model(data)
        loss = loss_function(embedding_a, embedding_b, data.y)
        loss.backward()
        total_loss += loss.item() * batch_size # last incomplete batch is dropped, so just use batch_size
        optimizer.step()
    train_loss = total_loss / train_pair_size
    return train_loss


def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    total_loss = 0
    for data in val_loader:
        data = data.to(device)
        embedding_a, embedding_b = model(data)
        loss = loss_function(embedding_a, embedding_b, data.y)

        # last incomplete batch is dropped, so just use batch_size
        total_loss += loss.item() * batch_size 

    val_loss = total_loss / val_size
    return val_loss


def validate_by_knn_acc():
    """
    Validate the training performance by k-nearest neighbor 
    accuracy on the validation set.
    """
    model.eval()

    # embeddings of train pockets
    train_embedding, train_label, _ = compute_embeddings(train_loader, model, device, normalize=True)

    # embeddings of validation pockets
    val_embedding, val_label, _ = compute_embeddings(val_loader, model, device, normalize=True)

    # knn model
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(train_embedding, train_label)
    train_prediction = knn.predict(train_embedding)
    val_prediction = knn.predict(val_embedding)
    train_acc = metrics.accuracy_score(train_label, train_prediction)
    val_acc = metrics.accuracy_score(val_label, val_prediction)

    return train_acc, val_acc


def compute_metrics(label, out):
    """
    Compute the evaluation metrics of the model.
    Both label and out should be converted from Pytorch tensor to numpy arrays containing 0s and 1s.
    """
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    return acc, precision, recall, f1, mcc


if __name__=="__main__":
    random.seed(666) # deterministic sampled pockets and pairs from dataset
    print('seed: ', 666)
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir
    subcluster_file = args.subcluster_file
    trained_model_dir = args.trained_model_dir
    loss_dir = args.loss_dir
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)    

    # number of clusters selected from the clusters
    num_classes = 14
    print('number of classes (from original clusters):', num_classes)

    # threshold of number of pockets in a class
    cluster_th = 10000 # large engouth to select all the data
    #print('max number of data of each class:', cluster_th)
    
    #merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, 10, 14, 15, 16, 17, 18]
    #merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, 10]
    merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8], 4, 6, 7, 10, 13]
    print('how to merge clusters: ', merge_info)

    subclustering = False # whether to further subcluster data according to subcluster_dict
    print('whether to further subcluster data according to chemical reaction: {}'.format( subclustering))

    train_pos_th = 16000 # threshold of number of positive train pairs for each class
    train_neg_th = 4600 # threshold of number of negative train pairs for each combination
    #val_pos_th = 3600 # threshold of number of positive validation pairs for each class
    #val_neg_th = 1100 # threshold of number of negative validation pairs for each combination
    print('positive training pair sampling threshold: ', train_pos_th)
    print('negative training pair sampling threshold: ', train_neg_th)
    #print('positive validation pair sampling threshold: ', val_pos_th)
    #print('negative validation pair sampling threshold: ', val_neg_th)

    # tunable hyper-parameters
    num_epochs = 55
    print('number of epochs to train:', num_epochs)
    lr_decay_epoch = 25
    print('learning rate decay to half at epoch {}.'.format(lr_decay_epoch))
    
    batch_size = 256
    print('batch size:', batch_size)
    learning_rate = 0.003
    weight_decay = 0.0005
    normalize = True # whether to normalize the embeddings in constrastive loss
    
    # margins for the relaxed contrastive loss
    similar_margin = 0.0
    dissimilar_margin = 2.0
    print('similar margin of contrastive loss: {}'.format(similar_margin))
    print('dissimilar margin of contrastive loss: {}'.format(dissimilar_margin))

    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    num_gpu = 0
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
    print('number of gpus: ', num_gpu)

    # missing popsa files for sasa feature at this moment
    #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'r', 'theta', 'phi', 'sasa', 'sequence_entropy'] 
    #features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'r', 'theta', 'phi', 'sequence_entropy'] 
    #features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'sasa', 'sequence_entropy'] 
    #features_to_use = ['x', 'y', 'z',  'r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
    #                   'binding_probability', 'sequence_entropy']
    features_to_use = ['r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
                       'binding_probability', 'sequence_entropy']
    print('features to use: ', features_to_use)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)    

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

    # train pairs
    train_pos_pairs, train_neg_pairs = gen_pairs(clusters=train_clusters, pos_pair_th=train_pos_th, neg_pair_th=train_neg_th)

    # validation pairs
    #val_pos_pairs, val_neg_pairs = gen_pairs(clusters=val_clusters, pos_pair_th=val_pos_th, neg_pair_th=val_neg_th)
    
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    train_pair_size = len(train_pos_pairs) +  len(train_neg_pairs)

    #print('number of validation positive pairs:', len(val_pos_pairs))
    #print('number of validation negative pairs:', len(val_neg_pairs))
    #val_size = len(val_pos_pairs) + len(val_neg_pairs)
    
    train_pair_loader = dataloader_gen(pocket_dir, 
                                       pop_dir,
                                       train_pos_pairs, 
                                       train_neg_pairs, 
                                       features_to_use, 
                                       batch_size, 
                                       shuffle=True,
                                       num_workers=num_workers)

    # dataloaders for validation 
    train_loader, train_size = pocket_loader_gen(pocket_dir=pocket_dir,
                                             pop_dir=pop_dir,
                                             clusters=train_clusters,
                                             features_to_use=features_to_use,
                                             batch_size=num_workers,
                                             shuffle=False,
                                             num_workers=num_workers)

    val_loader, val_size = pocket_loader_gen(pocket_dir=pocket_dir,
                                             pop_dir=pop_dir,
                                             clusters=val_clusters,
                                             features_to_use=features_to_use,
                                             batch_size=num_workers,
                                             shuffle=False,
                                             num_workers=num_workers)
    # end of dataloaders for validation

    #model = SiameseNet(num_features=len(features_to_use), dim=48, train_eps=True, num_edge_attr=1).to(device)
    model = ResidualSiameseNet(num_features=len(features_to_use), dim=48, train_eps=True, num_edge_attr=1).to(device)
    print('model architecture:')
    print(model)
    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    loss_function = ContrastiveLoss(similar_margin=similar_margin, dissimilar_margin=dissimilar_margin, normalize=normalize, mean=True).to(device) # differentiable, no parameters to train. 
    print('loss function:')
    print(loss_function)

    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    for epoch in range(1, num_epochs+1):
        train_loss = train()
        train_losses.append(train_loss)
        
        train_acc, val_acc = validate_by_knn_acc()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print('epoch: {}, train loss: {}, train acc: {}, validation acc: {}.'.format(epoch, train_loss, train_acc, val_acc))
        
        #if epoch > lr_decay_epoch: # store results for epochs after decay learning rate
        if  val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            torch.save(model.state_dict(), trained_model_dir)

    print('best validation acc {} at epoch {}.'.format(best_val_acc, best_val_epoch))

    # write loss history to disk
    results = {'train_losses': train_losses, 'train_accs': train_accs, 'val_accs': val_accs}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)
