import argparse
import os
import torch
from dataloader import divide_and_gen_pairs, dataloader_gen
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

    return parser.parse_args()


def train(batch_interval):
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    train_loss = []
    total_loss = 0
    for cnt, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        embedding_a, embedding_b = model(data)
        loss = loss_function(embedding_a, embedding_b, data.y)
        loss.backward()
        total_loss += loss.item() * batch_size # last incomplete batch is dropped, so just use batch_size
        optimizer.step()

        # log average loss every batch_interval batches
        if (cnt+1) % batch_interval == 0:
            train_loss.append(total_loss / (batch_size*batch_interval))
            total_loss = 0

    return train_loss

def validate(batch_interval):
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    val_loss = []
    total_loss = 0
    for cnt, data in enumerate(val_loader):
        data = data.to(device)
        embedding_a, embedding_b = model(data)
        loss = loss_function(embedding_a, embedding_b, data.y)
        total_loss += loss.item() * batch_size # last incomplete batch is dropped, so just use batch_size

        # log average loss every batch_interval batches
        if (cnt+1) % batch_interval == 0:
            val_loss.append(total_loss / (batch_size*batch_interval))
            total_loss = 0

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
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    num_classes = 50
    cluster_th = 400 # threshold of number of pockets in a class
    train_pos_th = 800 # threshold of number of positive train pairs
    train_neg_th = 40 # threshold of number of negative train pairs
    val_pos_th = 400 # threshold of number of positive validation pairs
    val_neg_th = 20 # threshold of number of negative validation pairs

    num_epochs = 200
    
    batch_size = 128
    print('batch size:', batch_size)
    
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    print('device: ', device)

    # missing popsa files for sasa feature at this moment
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 

    train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs, test_pos_pairs, test_neg_pairs = divide_and_gen_pairs(
                                                                                                    cluster_file_dir=cluster_file_dir, 
                                                                                                    num_classes=num_classes, 
                                                                                                    cluster_th=cluster_th,
                                                                                                    train_pos_th=train_pos_th,
                                                                                                    train_neg_th=train_neg_th,
                                                                                                    val_pos_th=val_pos_th,
                                                                                                    val_neg_th=val_neg_th)
    
    print('number of classes:', num_classes)
    print('max number of data of each class:', cluster_th)
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    train_size = len(train_pos_pairs) +  len(train_neg_pairs)

    print('number of validation positive pairs:', len(val_pos_pairs))
    print('number of validation negative pairs:', len(val_neg_pairs))
    val_size = len(val_pos_pairs) + len(val_neg_pairs)
    
    #print('number of test positive pairs:', len(test_pos_pairs))
    #print('number of test negative pairs:', len(test_neg_pairs))

    train_loader, val_loader, test_loader = dataloader_gen(pocket_dir, 
                                                           train_pos_pairs, 
                                                           train_neg_pairs, 
                                                           val_pos_pairs, 
                                                           val_neg_pairs, 
                                                           test_pos_pairs, 
                                                           test_neg_pairs, 
                                                           features_to_use, 
                                                           batch_size, 
                                                           shuffle=True,
                                                           num_workers=num_workers)

    model = SiameseNet(num_features=len(features_to_use), dim=32, train_eps=True, num_edge_attr=1).to(device)
    print('model architecture:')
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    loss_function = ContrastiveLoss(margin=2.0, normalize=True, mean=True) # differentiable, no parameters to train.
    print('loss function:')
    print(loss_function)

    print('number of epochs to train:', num_epochs)
    train_losses = []
    val_losses = []
    best_val_loss = 9999999
    for epoch in range(1, num_epochs+1):
        train_loss = train(batch_interval=50)
        train_losses.extend(train_loss)
        
        val_loss = validate(batch_interval=50) # list of losses every 50 mini-batches
        val_losses.extend(val_loss)

        epoch_loss = sum(val_loss) / len(val_loss)
        if  epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_val_epoch = epoch

    print('best validation loss {} at epoch {}.'.format(best_val_loss, best_val_epoch))

    results = {'train_losses': train_losses, 'val_losses': val_losses}
    result_dir = './train_results.json'

    with open(result_dir, 'w') as fp:
        json.dump(results, fp)