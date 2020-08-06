import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU
from torch_geometric.nn import GINConv, global_add_pool


class GINMolecularConv(GINConv):
    def __init__(self, nn, train_eps, num_features, num_edge_attr):
        """
        num_features: number of features of input nodes.
        """
        super(GINMolecularConv, self).__init__(nn=nn, train_eps=train_eps)
        self.edge_transformer = Sequential(Linear(num_edge_attr, num_features), 
                                           LeakyReLU(), 
                                           Linear(num_features, num_features),
                                           ELU()) # make it possible to reach -1

    def forward(self, x, edge_index, edge_attr, size = None):
        if isinstance(x, Tensor):
            x = (x, x) # x: OptPairTensor

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        weight = self.edge_transformer(edge_attr)
        return x_j * weight # element wise multiplication

    def __repr__(self):
        return '{}(nn={})(edge_transformer={})'.format(self.__class__.__name__, self.nn, self.edge_transformer)


class EmbeddingNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(EmbeddingNet, self).__init__()
        nn1 = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
        self.conv1 = GINMolecularConv(nn1, train_eps, num_features, num_edge_attr)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = GINMolecularConv(nn2, train_eps, dim, num_edge_attr)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv3 = GINMolecularConv(nn3, train_eps, dim, num_edge_attr)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv4 = GINMolecularConv(nn4, train_eps, dim, num_edge_attr)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        #nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv5 = GINConv(nn5)
        #self.bn5 = torch.nn.BatchNorm1d(dim)

        #nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv6 = GINConv(nn6)
        #self.bn6 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim) # generate embedding here

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        #x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)
        #x = F.relu(self.conv6(x, edge_index))
        #x = self.bn6(x)
        x = global_add_pool(x, batch)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        return x

    
class SiameseNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(num_features=num_features, dim=dim, train_eps=train_eps, num_edge_attr=num_edge_attr)

    def forward(self, pairdata):
        embedding_a = self.embedding_net(x=pairdata.x_a, edge_index=pairdata.edge_index_a, edge_attr=pairdata.edge_attr_a, batch=pairdata.x_a_batch)
        embedding_b = self.embedding_net(x=pairdata.x_b, edge_index=pairdata.edge_index_b, edge_attr=pairdata.edge_attr_b, batch=pairdata.x_b_batch)
        return embedding_a, embedding_b


class ContrastiveLoss(torch.nn.Module):
    """
    ContrastiveLoss introduced in the paper "Dimensionality Reduction by Learning an Invariant Mapping".
    We add L2 normalizations to constrain the input embeddings to be on a hypersphere of radius 1. 
    The purpose of adding the normalization is for easy choice of the margin hyper-parameter.
    """
    def __init__(self, margin=2.0, normalize=True, mean=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin # the margin in the paper
        self.normalize = normalize # whether to normalize input embeddings
        self.mean = mean # mean over batch or sum over batch

    def forward(self, embedding_a, embedding_b, label):
        """
        label = 1 for similar pairs, 0 for dissimilar pairs.
        """
        if self.normalize == True:
            embedding_a = F.normalize(embedding_a)
            embedding_b = F.normalize(embedding_b)
        euclidean_dist = F.pairwise_distance(embedding_a, embedding_b)
        ls = label * torch.pow(euclidean_dist, 2) # loss for similar pairs

        ld = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0), 2) # loss for dissimilar pairs 

        loss = ls + ld 

        if self.mean == True:
            return loss.mean()
        else:
            return loss.sum()

    def __repr__(self):
        return 'ContrastiveLoss(margin={}, normalize={}, mean={})'.format(self.margin, self.normalize, self.mean)


# TO-DO: logistic similarity for cross-entropy loss

# TO-DO: similarity by dot product of node features


if __name__=="__main__":
    """
    Main function for testing and debugging only
    """
    a = torch.tensor([[0,0,1],[1,2,3]], dtype=torch.float32)
    b = torch.tensor([[0,0,-1],[1,2,3]], dtype=torch.float32)
    y = torch.tensor([0], dtype=torch.long)
    loss=ContrastiveLoss()
    print(loss(a,b,y))

