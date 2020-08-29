import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import Set2Set


class GINMolecularConv(GINConv):
    def __init__(self, nn, train_eps, num_features, num_edge_attr):
        """
        num_features: number of features of input nodes.
        """
        super(GINMolecularConv, self).__init__(nn=nn, train_eps=train_eps)
        self.edge_transformer = Sequential(Linear(num_edge_attr, 8), 
                                           LeakyReLU(), 
                                           Linear(8, num_features),
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
        # neural network to compute self-attention for output layer
        #gate_nn = Sequential(Linear(dim+num_features, dim+num_features), LeakyReLU(), Linear(dim+num_features, 1), LeakyReLU())
        # neural netowrk to compute embedding before masking
        #out_nn =  Sequential(Linear(dim+num_features, dim+num_features), LeakyReLU(), Linear(dim+num_features, dim+num_features)) # followed by softmax
        # global attention pooling layer
        #self.global_att = GlobalAttention(gate_nn=gate_nn, nn=out_nn)
        self.set2set = Set2Set(in_channels=dim, processing_steps=5, num_layers=2)

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

        #self.fc1 = Linear(dim, dim)
        #self.fc2 = Linear(dim, dim) # generate embedding here

    def forward(self, x, edge_index, edge_attr, batch):
        #x_in = x 
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)

        #x = global_add_pool(x, batch)
        #x = self.global_att(torch.cat((x, x_in), 1), batch)
        x = self.set2set(x, batch)
        return x

    
class SiameseNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet(num_features=num_features, dim=dim, train_eps=train_eps, num_edge_attr=num_edge_attr)

    def forward(self, pairdata):
        embedding_a = self.embedding_net(x=pairdata.x_a, edge_index=pairdata.edge_index_a, edge_attr=pairdata.edge_attr_a, batch=pairdata.x_a_batch)
        embedding_b = self.embedding_net(x=pairdata.x_b, edge_index=pairdata.edge_index_b, edge_attr=pairdata.edge_attr_b, batch=pairdata.x_b_batch)
        return embedding_a, embedding_b

    def get_embedding(self, data, normalize):
        """
        Used to get the embedding of a pocket after training.
        data: standard PyG graph data.
        """
        embedding = self.embedding_net(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        
        # normalize the embedding if the embeddings are normalized during training
        # see ContrastiveLoss.__init__()
        if normalize == True:
            embedding = F.normalize(embedding)
        
        return embedding
        

class ResidualBlock(torch.nn.Module):
    """
    A residual block which has two graph neural network layers. The output and input are summed 
    so that the module can learn identity function.
    """
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(ResidualBlock, self).__init__()
        
        nn1 = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
        self.conv1 = GINMolecularConv(nn1, train_eps, num_features, num_edge_attr)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = GINMolecularConv(nn2, train_eps, dim, num_edge_attr)
        
        self.bn = torch.nn.BatchNorm1d(dim)
    
    def forward(self, x, edge_index, edge_attr):
        x_skip = x # store the input value
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = x + x_skip # add before activation
        x = F.leaky_relu(x)
        x = self.bn(x)
        return x


class ResidualEmbeddingNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(ResidualEmbeddingNet, self).__init__()

        # first graph convolution layer, increasing dimention
        nn1 = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
        self.conv1 = GINMolecularConv(nn1, train_eps, num_features, num_edge_attr)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        # residual blocks
        self.rb_2 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_3 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_4 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_5 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_6 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_7 = ResidualBlock(dim, dim, train_eps, num_edge_attr)

        # read out function
        self.set2set = Set2Set(in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)

        x = self.rb_2(x, edge_index, edge_attr)
        x = self.rb_3(x, edge_index, edge_attr)
        x = self.rb_4(x, edge_index, edge_attr)
        x = self.rb_5(x, edge_index, edge_attr)
        x = self.rb_6(x, edge_index, edge_attr)
        x = self.rb_7(x, edge_index, edge_attr)

        #x = global_add_pool(x, batch)
        #x = self.global_att(torch.cat((x, x_in), 1), batch)
        x = self.set2set(x, batch)
        return x


class ResidualSiameseNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(ResidualSiameseNet, self).__init__()
        self.embedding_net = ResidualEmbeddingNet(num_features=num_features, dim=dim, train_eps=train_eps, num_edge_attr=num_edge_attr)

    def forward(self, pairdata):
        embedding_a = self.embedding_net(x=pairdata.x_a, edge_index=pairdata.edge_index_a, edge_attr=pairdata.edge_attr_a, batch=pairdata.x_a_batch)
        embedding_b = self.embedding_net(x=pairdata.x_b, edge_index=pairdata.edge_index_b, edge_attr=pairdata.edge_attr_b, batch=pairdata.x_b_batch)
        return embedding_a, embedding_b

    def get_embedding(self, data, normalize):
        """
        Used to get the embedding of a pocket after training.
        data: standard PyG graph data.
        """
        embedding = self.embedding_net(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        
        # normalize the embedding if the embeddings are normalized during training
        # see ContrastiveLoss.__init__()
        if normalize == True:
            embedding = F.normalize(embedding)
        
        return embedding


class ContrastiveLoss(torch.nn.Module):
    """
    ContrastiveLoss introduced in the paper "Dimensionality Reduction by Learning an Invariant Mapping".
    We add L2 normalizations to constrain the input embeddings to be on a hypersphere of radius 1. 
    The purpose of adding the normalization is for easy choice of the margins hyper-parameter.

    The loss can be relaxed by setting the margins. The intuition is that similar pockets do not need 
    to have identical embeddings. 
    """
    def __init__(self, similar_margin=0.0, dissimilar_margin=2.0, normalize=True, mean=True):
        super(ContrastiveLoss, self).__init__()
        self.similar_margin = similar_margin
        self.dissimilar_margin = dissimilar_margin # the margin in original paper for dissimilar pairs
        self.normalize = normalize # whether to normalize input embeddings
        self.mean = mean # mean over batch or sum over batch

    def forward(self, embedding_a, embedding_b, label):
        """
        label = 1 for similar pairs, 0 for dissimilar pairs.
        """
        if self.normalize == True:
            embedding_a = F.normalize(embedding_a)
            embedding_b = F.normalize(embedding_b)

        # distance between the pairs
        euclidean_dist = F.pairwise_distance(embedding_a, embedding_b)

        # loss for similar pairs
        #ls = label * torch.pow(euclidean_dist, 2)
        ls = label * torch.pow(torch.clamp(euclidean_dist - self.similar_margin, min=0), 2)
        
        # loss for dissimilar pairs 
        ld = (1 - label) * torch.pow(torch.clamp(self.dissimilar_margin - euclidean_dist, min=0), 2) 

        loss = ls + ld 

        if self.mean == True:
            return loss.mean()
        else:
            return loss.sum()

    def __repr__(self):
        return 'ContrastiveLoss(similar_margin={}, dissimilar_margin={}, normalize={}, mean={})'.format(self.similar_margin, self.dissimilar_margin, self.normalize, self.mean)


class MoNet(torch.nn.Module):
    """Standard classifier to solve the problem.""" 
    def __init__(self, num_classes, num_features, dim, train_eps, num_edge_attr):
        super(MoNet, self).__init__()
        self.num_classes = num_classes

        # neural network to compute self-attention for output layer
        #gate_nn = Sequential(Linear(dim+num_features, dim+num_features), LeakyReLU(), Linear(dim+num_features, 1), LeakyReLU())
        
        # neural netowrk to compute embedding before masking
        #out_nn =  Sequential(Linear(dim+num_features, dim+num_features), LeakyReLU(), Linear(dim+num_features, dim+num_features)) # followed by softmax
        
        # global attention pooling layer
        #self.global_att = GlobalAttention(gate_nn=gate_nn, nn=out_nn)
        
        self.set2set = Set2Set(in_channels=dim, processing_steps=5, num_layers=2)

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

        self.fc1 = Linear(2 * dim, self.num_classes)
        #self.fc2 = Linear(dim, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # x_in = x 
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)

        x_skip = x

        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)

        x = x + x_skip

        #x = self.global_att(torch.cat((x, x_in), 1), batch)
        #x = global_add_pool(x, batch)
        x = self.set2set(x, batch)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)

        # x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# TO-DO: functions to accumulate/reset ls and ld for analysis.

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

