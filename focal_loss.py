
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, mean=True, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mean= mean
        self.size_average = size_average
        self.epsilon = 0.00000001
    
    def forward(self, logits, labels):
        """
        calculates loss
        logits: output by model, size: [batch_size, num_class]
        labels: groud truth of classification label, size: [batch_size]
        """
        print(F.softmax(logits, dim=-1))
        print(-1 * F.log_softmax(logits, dim=-1))
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        print(ce_loss)
        pt = torch.exp(-ce_loss)
        print(pt)

        '''
        logits = F.log_softmax(logits, dim=-1)
        print(logits)
        batch_size = logits.shape[0]
        print(batch_size)
        pt = torch.zeros([batch_size])
        print(pt)
        '''
        return None




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
