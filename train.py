import argparse
import random
import os
import torch
from torch_geometric.nn import DataParallel
from dataloader import divide_and_gen_pairs, dataloader_gen, dataloader_gen_multi_gpu
from model import SiameseNet, ContrastiveLoss
import sklearn.metrics as metrics
import json


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
    Global vars: train_loader, train_size, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding_a, embedding_b = model(data)
        loss = loss_function(embedding_a, embedding_b, data.y)
        loss.backward()
        total_loss += loss.item() * batch_size # last incomplete batch is dropped, so just use batch_size
        optimizer.step()
    train_loss = total_loss / train_size
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
        total_loss += loss.item() * batch_size # last incomplete batch is dropped, so just use batch_size

    val_loss = total_loss / val_size
    return val_loss


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
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    trained_model_dir = args.trained_model_dir
    loss_dir = args.loss_dir
    
    num_classes = 60
    print('number of classes:', num_classes)
    cluster_th = 10000 # threshold of number of pockets in a class
    #print('max number of data of each class:', cluster_th)
    
    train_pos_th = 3000 # threshold of number of positive train pairs for each class
    train_neg_th = 100 # threshold of number of negative train pairs for each combination
    val_pos_th = 1000 # threshold of number of positive validation pairs for each class
    val_neg_th = 25 # threshold of number of negative validation pairs for each combination

    # tunable hyper-parameters
    num_epochs = 50
    print('number of epochs to train:', num_epochs)
    batch_size = 256
    print('batch size:', batch_size)
    learning_rate = 0.003
    weight_decay = 0.0005
    
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
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 

    train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs = divide_and_gen_pairs(cluster_file_dir=cluster_file_dir, 
                                                                                          num_classes=num_classes, 
                                                                                          cluster_th=cluster_th,
                                                                                          train_pos_th=train_pos_th,
                                                                                          train_neg_th=train_neg_th,
                                                                                          val_pos_th=val_pos_th,
                                                                                          val_neg_th=val_neg_th)
    
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    train_size = len(train_pos_pairs) +  len(train_neg_pairs)

    print('number of validation positive pairs:', len(val_pos_pairs))
    print('number of validation negative pairs:', len(val_neg_pairs))
    val_size = len(val_pos_pairs) + len(val_neg_pairs)
    
    train_loader, val_loader = dataloader_gen(pocket_dir, 
                                                           train_pos_pairs, 
                                                           train_neg_pairs, 
                                                           val_pos_pairs, 
                                                           val_neg_pairs, 
                                                           features_to_use, 
                                                           batch_size, 
                                                           shuffle=True,
                                                           num_workers=num_workers)

    model = SiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)
    print('model architecture:')
    print(model)
    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    loss_function = ContrastiveLoss(margin=2.0, normalize=False, mean=True).to(device) # differentiable, no parameters to train. 
    print('loss function:')
    print(loss_function)

    train_losses = []
    val_losses = []
    best_val_loss = 9999999
    for epoch in range(1, num_epochs+1):
        train_loss = train()
        train_losses.append(train_loss)
        
        val_loss = validate() # list of losses every 50 mini-batches
        val_losses.append(val_loss)

        print('train loss: {}, validation loss: {}.'.format(train_loss, val_loss))

        if  val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), trained_model_dir)

    print('best validation loss {} at epoch {}.'.format(best_val_loss, best_val_epoch))

    results = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

# TO-DO: multi-GPU support