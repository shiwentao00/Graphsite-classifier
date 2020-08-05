import argparse
import multiprocessing
from dataloader import dataloader_gen
from model import SiameseNet, ContrastiveLoss


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

    parser.add_argument('-batch_size',
                        type=int,
                        default=32,
                        required=False,
                        help='size of mini-batches.')

    return parser.parse_args()


def train(epoch):
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    for data in train_loader:
        #print('data.x: ', data.x)
        #print('data.edge_index: ', data.edge_index)
        #print(data.y)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        #print('gradients of conv1 nn[0]:', model.conv1.nn[0].weight.grad)
        #print('gradients of conv1 nn[2]:', model.conv1.nn[2].weight.grad)
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
        pred = output.max(dim=1)[1]
        output_prob = torch.exp(output) # output probabilities of each class
    
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics
        
        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    acc, precision, recall, f1, mcc = compute_metrics(epoch_label, epoch_pred)
    train_loss = loss_total / train_size # averaged training loss
    result_dict = {'acc':acc, 'precision': precision, 'recall': recall, 'f1':f1, 'mcc': mcc, 'loss': train_loss}   
    return result_dict

def validate():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0
    epoch_pred = [] # all the predictions for the epoch
    epoch_label = [] # all the labels for the epoch
    epoch_prob = [] # all the output probabilities
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.nll_loss(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]
        output_prob = torch.exp(output) # output probabilities of each class
        
        pred_cpu = list(pred.cpu().detach().numpy()) # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy()) # used to compute evaluation metrics
        output_prob_cpu = output_prob.cpu().detach().numpy() # softmax output
        output_prob_cpu = list(output_prob_cpu[:,1]) # probability of positive class

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)
        epoch_prob.extend(output_prob_cpu)
        
    val_loss = loss_total / val_size # averaged training loss
    acc, precision, recall, f1, mcc = compute_metrics(epoch_label, epoch_pred) # evaluation metrics
    result_dict = {'acc':acc, 'precision': precision, 'recall': recall, 'f1':f1, 'mcc': mcc, 'loss': val_loss}   
    roc_dict = {'label':epoch_label, 'prob': epoch_prob}    # data needed to compute roc curve 
    return result_dict, roc_dict


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
    batch_size = args.batch_size
    num_workers = multiprocessing.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    num_classes = 150
    cluster_th = 400

    # missing popsa files for sasa feature at this moment
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sequence_entropy'] 

    train_pos_pairs, 
    train_neg_pairs, 
    val_pos_pairs, 
    val_neg_pairs, 
    test_pos_pairs, 
    test_neg_pairs = divide_and_gen_pairs(cluster_file_dir=cluster_file_dir, num_classes=num_classes, cluster_th=cluster_th)
    

    print('number of classes:', num_classes)
    print('max number of data of each class:', cluster_th)
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    print('number of validation positive pairs:', len(val_pos_pairs))
    print('number of validation negative pairs:', len(val_neg_pairs))
    print('number of test positive pairs:', len(test_pos_pairs))
    print('number of test negative pairs:', len(test_neg_pairs))

    tarin_loader, val_loader, test_loader = dataloader_gen(pocket_dir, 
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