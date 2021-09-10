# Copyright: Wentao Shi, 2020
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import GINConv
from torch_geometric.nn import PNAConv, BatchNorm
from torch_geometric.nn import Set2Set
import itertools
import numpy as np
from torch_geometric.nn import MessagePassing


class SCNWMConv(GINConv):
    """
    This model implements single-channel neural weighted message. It can be
    done by inheriting the CINConv and adding the edge neural network. 
    """

    def __init__(self, nn, train_eps, num_features, num_edge_attr):
        """
        num_features: number of features of input nodes.
        """
        super(SCNWMConv, self).__init__(nn=nn, train_eps=train_eps)
        self.edge_transformer = Sequential(Linear(num_edge_attr, 8),
                                           LeakyReLU(),
                                           Linear(8, 1),
                                           ELU()
                                           )

    def forward(self, x, edge_index, edge_attr, size=None):
        # x: OptPairTensor
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        weight = self.edge_transformer(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node share one weight generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(nn={})(edge_transformer={})'.format(
            self.__class__.__name__,
            self.nn,
            self.edge_transformer
        )


class EmbeddingNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(EmbeddingNet, self).__init__()

        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2
        )

        nn1 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv1 = SCNWMConv(nn1, train_eps, num_features, num_edge_attr)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = SCNWMConv(nn2, train_eps, dim, num_edge_attr)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv3 = SCNWMConv(nn3, train_eps, dim, num_edge_attr)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv4 = SCNWMConv(nn4, train_eps, dim, num_edge_attr)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        #nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv5 = GINConv(nn5)
        #self.bn5 = torch.nn.BatchNorm1d(dim)

        #nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv6 = GINConv(nn6)
        #self.bn6 = torch.nn.BatchNorm1d(dim)

        #self.fc1 = Linear(dim, dim)
        # self.fc2 = Linear(dim, dim) # generate embedding here

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)

        x = self.set2set(x, batch)
        return x


class ResidualBlock(torch.nn.Module):
    """
    A residual block which has two graph neural network layers. The output and input are summed 
    so that the module can learn identity function.
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(ResidualBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm1d(dim)
        nn1 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv1 = SCNWMConv(nn1, train_eps, num_features, num_edge_attr)

    def forward(self, x, edge_index, edge_attr):
        # store the input value
        x_skip = x

        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv1(x, edge_index, edge_attr)

        # add before activation
        x = x + x_skip

        return x


class ResidualEmbeddingNet(torch.nn.Module):
    """
    Embedding network with residual connections.
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(ResidualEmbeddingNet, self).__init__()

        # first graph convolution layer, increasing dimention
        nn1 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv1 = SCNWMConv(nn1, train_eps, num_features, num_edge_attr)

        # residual blocks
        self.rb_2 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_3 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_4 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_5 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_6 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_7 = ResidualBlock(dim, dim, train_eps, num_edge_attr)
        self.rb_8 = ResidualBlock(dim, dim, train_eps, num_edge_attr)

        # batch norm for last conv layer
        self.bn_8 = torch.nn.BatchNorm1d(dim)

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)

        x = self.rb_2(x, edge_index, edge_attr)
        x = self.rb_3(x, edge_index, edge_attr)
        x = self.rb_4(x, edge_index, edge_attr)
        x = self.rb_5(x, edge_index, edge_attr)
        x = self.rb_6(x, edge_index, edge_attr)
        x = self.rb_7(x, edge_index, edge_attr)
        x = self.rb_8(x, edge_index, edge_attr)

        # batch norm after activation
        x = F.leaky_relu(x)
        x = self.bn_8(x)

        x = self.set2set(x, batch)

        return x


class JKEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on 
    Graphs with Jumping Knowledge Networks".

    This model uses single-channle neural message masking (SCNWMConv module).
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr,
                 num_layers, layer_aggregate='max'):
        super(JKEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        nn0 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv0 = SCNWMConv(nn0, train_eps, num_features, num_edge_attr)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec(
                'nn{} = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))'.format(i))
            exec(
                'self.conv{} = SCNWMConv(nn{}, train_eps, dim, num_edge_attr)'.format(i, i))
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)
        return x


class PNAEmbeddingNet(torch.nn.Module):
    """
    EmbeddingNet with PNAConv layers from the paper "Principal 
    Neighbourhood Aggregation for Graph Nets".
    """

    def __init__(self, num_features, dim, num_edge_attr, num_layers, deg):
        super(PNAEmbeddingNet, self).__init__()

        # define the aggregators and scalers, can be more
        #aggregators = ['mean', 'min', 'max', 'std']
        #scalers = ['identity', 'amplification', 'attenuation']
        aggregators = ['mean', 'min', 'max']
        scalers = ['identity']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        # first layer
        conv0 = PNAConv(
            in_channels=num_features, out_channels=dim,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=num_edge_attr, towers=4, pre_layers=1, post_layers=1,
            divide_input=False
        )
        bn0 = BatchNorm(dim)
        self.convs.append(conv0)
        self.batch_norms.append(bn0)

        # rest layers
        for _ in range(1, num_layers):
            conv = PNAConv(
                in_channels=dim, out_channels=dim,
                aggregators=aggregators, scalers=scalers, deg=deg,
                edge_dim=num_edge_attr, towers=4, pre_layers=1, post_layers=1,
                divide_input=False
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(dim))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # jumping knowledge
        layer_x = []
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.leaky_relu(batch_norm(conv(x, edge_index, edge_attr)))
            layer_x.append(x)

        # layer aggregation
        x = torch.stack(layer_x, dim=0)
        x = torch.max(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)

        return x


class NWMConv(MessagePassing):
    """
    The neural weighted message (NWM) layer. output of multiple instances of this
    will produce multi-channel output. 
    """

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(Linear(num_edge_attr, 8),
                                  LeakyReLU(),
                                  Linear(8, 1),
                                  ELU())
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr, size=None):
        # x: OptPairTensor
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(edge_nn={})'.format(self.__class__.__name__, self.edge_nn)


class MCNWMConv(torch.nn.Module):
    """
    Multi-channel neural weighted message module.
    """

    def __init__(self, in_dim, out_dim, num_channels,
                 num_edge_attr=1, train_eps=True, eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(Linear(in_dim * num_channels,
                                    out_dim), LeakyReLU(), Linear(out_dim, out_dim))
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(
                nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x


class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on 
    Graphs with Jumping Knowledge Networks".

    The GNN layers are now MCNWMConv layer
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr,
                 num_layers, num_channels=1, layer_aggregate='max'):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        self.conv0 = MCNWMConv(in_dim=num_features, out_dim=dim, num_channels=num_channels,
                               num_edge_attr=num_edge_attr, train_eps=train_eps)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec(
                'self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)'.format(
                    i, num_channels)
            )
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)

        return x


class JKEGINEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on 
    Graphs with Jumping Knowledge Networks".

    The layer model is GIN, which does not take edge attribute as input. This is used as
    the baseline model in the paper.
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr,
                 num_layers, layer_aggregate='max'):
        super(JKEGINEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        nn0 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv0 = GINConv(nn=nn0, train_eps=train_eps)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec(
                'nn{} = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))'.format(i))
            exec('self.conv{} = GINConv(nn=nn{}, train_eps=train_eps)'.format(i, i))
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)
        return x


class GraphSiteClassifier(torch.nn.Module):
    """Standard classifier to classify the binding sites."""

    def __init__(self, num_classes, num_features, dim, train_eps,
                 num_edge_attr, which_model, num_layers, num_channels,
                 deg=None):
        """
        train_eps: for the SCNWMConv module only when which_model in 
        ['jk', 'residual', 'jknmm', and 'normal'].
        deg: for PNAEmbeddingNet only, can not be None when which_model=='pna'.
        """
        super(GraphSiteClassifier, self).__init__()
        self.num_classes = num_classes

        # use one of the embedding net
        if which_model == 'residual':
            self.embedding_net = ResidualEmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr
            )
        elif which_model == 'jk':
            self.embedding_net = JKEmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr,
                num_layers=num_layers
            )
        elif which_model == 'pna':
            self.embedding_net = PNAEmbeddingNet(
                num_features=num_features,
                dim=dim, num_edge_attr=num_edge_attr,
                num_layers=num_layers, deg=deg
            )
        elif which_model == 'jknwm':
            self.embedding_net = JKMCNWMEmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr,
                num_layers=num_layers,
                num_channels=num_channels
            )
        elif which_model == 'jkgin':
            self.embedding_net = JKEGINEmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr,
                num_layers=num_layers
            )
        else:
            self.embedding_net = EmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr)

        # set2set doubles the size of embeddnig
        self.fc1 = Linear(2 * dim, dim)
        self.fc2 = Linear(dim, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_net(x=x, edge_index=edge_index,
                               edge_attr=edge_attr, batch=batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        # returned tensor should be processed by a softmax layer
        return x


class FocalLoss(torch.nn.Module):
    """
    Implement the Focal Loss introduced in the paper "Focal Loss for Dense Object Detection"
    """

    def __init__(self, gamma=2, alpha=1, reduction='mean'):
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
        # probablities of all classes generated by softmax
        softmax_pt = F.softmax(logits, dim=-1)

        # select softmax probablity according to label
        # this is the 'pt' in the paper
        pt = torch.squeeze(softmax_pt.gather(1, labels.view(-1, 1)))

        # take the negative log to compute cross entropy
        ce_loss = (-1) * torch.log(pt)

        # select the alpha, i.e., the class weight according to label
        alpha = self.alpha[labels]

        # rescale the cross entropy loss
        focal_loss = alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()

    def set_gamma(self, gamma):
        """
        Change the value of gamma during training.
        """
        self.gamma = gamma

    def __repr__(self):
        return '{}(gamma={}, alpha={}, reduction={})'.format(
            self.__class__.__name__,
            self.gamma, self.alpha,
            self.reduction
        )


class SiameseNet(torch.nn.Module):
    """
    SiameseNet with 3 choices of architectures
    """

    def __init__(self, num_features, dim, train_eps,
                 num_edge_attr, which_model, num_layers, deg=None):
        super(SiameseNet, self).__init__()
        if which_model == 'residual':
            self.embedding_net = ResidualEmbeddingNet(
                num_features=num_features, dim=dim,
                train_eps=train_eps, num_edge_attr=num_edge_attr
            )
        elif which_model == 'jk':
            self.embedding_net = JKEmbeddingNet(
                num_features=num_features, dim=dim,
                train_eps=train_eps, num_edge_attr=num_edge_attr, num_layers=num_layers
            )
        else:
            self.embedding_net = EmbeddingNet(
                num_features=num_features, dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr
            )

    def forward(self, pairdata):
        embedding_a = self.embedding_net(
            x=pairdata.x_a, edge_index=pairdata.edge_index_a,
            edge_attr=pairdata.edge_attr_a, batch=pairdata.x_a_batch
        )
        embedding_b = self.embedding_net(
            x=pairdata.x_b, edge_index=pairdata.edge_index_b,
            edge_attr=pairdata.edge_attr_b, batch=pairdata.x_b_batch
        )
        return embedding_a, embedding_b

    def get_embedding(self, data, normalize):
        """
        Used to get the embedding of a pocket after training.
        data: standard PyG graph data.
        """
        embedding = self.embedding_net(
            x=data.x, edge_index=data.edge_index,
            edge_attr=data.edge_attr, batch=data.batch
        )

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
        # the margin in original paper for dissimilar pairs
        self.dissimilar_margin = dissimilar_margin
        self.normalize = normalize  # whether to normalize input embeddings
        self.mean = mean  # mean over batch or sum over batch

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
        ls = label * \
            torch.pow(torch.clamp(euclidean_dist -
                                  self.similar_margin, min=0), 2)

        # loss for dissimilar pairs
        ld = (1 - label) * torch.pow(torch.clamp(self.dissimilar_margin -
                                                 euclidean_dist, min=0), 2)

        loss = ls + ld

        if self.mean == True:
            return loss.mean()
        else:
            return loss.sum()

    def __repr__(self):
        return 'ContrastiveLoss(similar_margin={}, dissimilar_margin={}, normalize={}, mean={})'.format(
            self.similar_margin,
            self.dissimilar_margin,
            self.normalize,
            self.mean
        )


class SelectiveSiameseNet(torch.nn.Module):
    """
    SiameseNet model that used with the SelectiveContrastiveLoss. It is a module
    wrapping a EmbeddingNet with a 'get_embedding' class method.   

    For this model, only the hardest pairs in a mini-batch are selected dynamically 
    for training and validation.
    """

    def __init__(self, num_features, dim, train_eps,
                 num_edge_attr, which_model, num_layers, deg=None):
        super(SelectiveSiameseNet, self).__init__()
        if which_model == 'residual':
            self.embedding_net = ResidualEmbeddingNet(
                num_features=num_features, dim=dim,
                train_eps=train_eps, num_edge_attr=num_edge_attr
            )
        elif which_model == 'jk':
            self.embedding_net = JKEmbeddingNet(
                num_features=num_features, dim=dim,
                train_eps=train_eps, num_edge_attr=num_edge_attr,
                num_layers=num_layers
            )
        else:
            self.embedding_net = EmbeddingNet(
                num_features=num_features, dim=dim,
                train_eps=train_eps, num_edge_attr=num_edge_attr
            )

    def forward(self, data):
        embedding = self.embedding_net(
            x=data.x, edge_index=data.edge_index,
            edge_attr=data.edge_attr, batch=data.batch
        )

        # normalize the embedding to force them to sit on a hyper-sphere.
        embedding = F.normalize(embedding)

        return embedding

    def get_embedding(self, data, normalize=True):
        """
        Same function as the forward function. Used to get the embedding of a pocket after training.

        Argument:
        data - standard PyG graph data.
        normalize - must be set to true. 
        """
        embedding = self.embedding_net(
            x=data.x, edge_index=data.edge_index,
            edge_attr=data.edge_attr, batch=data.batch
        )

        # normalize the embedding if the embeddings are normalized during training
        # see ContrastiveLoss.__init__()
        if normalize == True:
            embedding = F.normalize(embedding)

        return embedding


class SelectiveContrastiveLoss(torch.nn.Module):
    """
    The contrastive loss that selects hard pairs to optimize.
    """

    def __init__(self, similar_margin=0.0, dissimilar_margin=2.0,
                 num_pos_pair=128, num_neg_pair=128):
        super(SelectiveContrastiveLoss, self).__init__()
        self.similar_margin = similar_margin
        self.dissimilar_margin = dissimilar_margin

        # the max number of postive pairs to send to loss
        self.num_pos_pair = num_pos_pair

        # the max number of negative pairs to send to loss
        self.num_neg_pair = num_neg_pair

        # the switch to control how the pairs are selected
        # if True, select hardest pairs to optimize
        # if False, select random pairs to optimize
        self.select_hard_pairs = False

    def forward(self, embedding, label):
        #pos_pairs = self.__select_pos_pair(embedding, label)
        #neg_pairs = self.__select_neg_pair(embedding, label)
        label = label.cpu().detach().numpy()

        # all possible pairs of index
        pairs = np.array(list(itertools.combinations(range(len(label)), 2)))

        pos_pair_idx = pairs[np.nonzero(
            label[pairs[:, 0]] == label[pairs[:, 1]])[0], :]
        neg_pair_idx = pairs[np.nonzero(
            label[pairs[:, 0]] != label[pairs[:, 1]])[0], :]

        # compute loss for similar (positive) and dissimilar (negative) pairs separately
        similar_loss = self.__compute_similar_loss(
            embedding, label, pos_pair_idx, self.num_pos_pair)
        dissimilar_loss = self.__compute_dissimilar_loss(
            embedding, label, neg_pair_idx, self.num_neg_pair)

        # the program is guaranteed to generate positive pairs, or error will be raised
        if dissimilar_loss is None:
            loss = similar_loss
        else:
            loss = torch.cat([similar_loss, dissimilar_loss])

        # mean for back propagation, sum for logging
        return loss.mean(), loss.sum(), loss.shape

    def __compute_similar_loss(self, embedding, label, pos_pair_idx, num_pairs):
        """Get all the positive pairs and compute the loss"""
        # compute the number of pairs sent to the loss
        total_num_pairs = pos_pair_idx.shape[0]
        num_pairs = min(num_pairs, total_num_pairs)
        #print('total number of positive pairs: ', total_num_pairs)
        #print('actual number of positive pairs sent to loss: ', num_pairs)

        # no loss if there is no positive pairs
        if total_num_pairs == 0:
            raise ValueError('No similar pairs, increase the batch size.')

        # select embedding of positive pairs
        embedding_a = embedding[pos_pair_idx[:, 0]]
        embedding_b = embedding[pos_pair_idx[:, 1]]

        # compute the loss
        euclidean_dist = F.pairwise_distance(embedding_a, embedding_b)
        loss = torch.pow(torch.clamp(euclidean_dist -
                                     self.similar_margin, min=0), 2)
        if self.select_hard_pairs == True:
            loss, _ = torch.sort(loss, descending=True)
        loss = loss[0: num_pairs]  # select top num_pairs loss

        return loss

    def __compute_dissimilar_loss(self, embedding, label, neg_pair_idx, num_pairs):
        """Select the most dissimilar pairs in the mini-batch and compute the loss"""
        # compute the number of pairs sent to the loss
        total_num_pairs = neg_pair_idx.shape[0]
        num_pairs = min(num_pairs, total_num_pairs)
        #print('total number of negative pairs: ', total_num_pairs)
        #print('actual number of negative pairs sent to loss: ', num_pairs)

        # no loss if there is no negative pairs
        if total_num_pairs == 0:
            return None

        # select embedding of negative pairs
        embedding_a = embedding[neg_pair_idx[:, 0]]
        embedding_b = embedding[neg_pair_idx[:, 1]]

        # compute the loss
        euclidean_dist = F.pairwise_distance(embedding_a, embedding_b)
        loss = torch.pow(torch.clamp(
            self.dissimilar_margin - euclidean_dist, min=0), 2)
        if self.select_hard_pairs == True:
            loss, _ = torch.sort(loss, descending=True)
        loss = loss[0: num_pairs]  # select top num_pairs loss

        return loss

    def set_select_hard_pairs(self, select_hard_pairs):
        """used to alternate the pair selection during training"""
        assert(select_hard_pairs in [True, False])
        self.select_hard_pairs = select_hard_pairs

    def __repr__(self):
        return 'SelectiveContrastiveLoss(similar_margin={}, dissimilar_margin={}, num_pos_pair={}, num_neg_pair={})'.format(
            self.similar_margin, self.dissimilar_margin, self.num_pos_pair, self.num_neg_pair
        )
