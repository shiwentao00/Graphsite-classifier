
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss

class FocalLoss(torch.nn.Module):
    """
    Implement the Focal Loss introduced in the paper "Focal Loss for Dense Object Detection"
    """
    def __init__(self, gamma = 2, alpha = 1, reduction='mean'):
        """
        gamma: the modulation factor in the paper.
        alpha: class weights, default is scalar 1 which means equal weights for all instances.
               Can also be a tensor with size num_class
        mean: reduction method, 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        logits: output by model, size: [batch_size, num_class].
        labels: groud truth of classification label, size: [batch_size].
        """
        # cross entropy loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # reversely compute the softmax probability
        pt = torch.exp(-ce_loss)

        # rescale the cross entropy loss
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()


if __name__ == "__main__":
    logits = torch.rand(5, 8) # batch size 5, 8 classes
    labels = torch.LongTensor([0,1,2,3,4])
    print(logits.shape)
    print(labels.shape)

    #cel = CrossEntropyLoss()
    #cel_loss = cel(logits, labels)
    #print(cel_loss)

    fl = FocalLoss()
    fl_loss = fl(logits, labels)
    print(fl_loss)
