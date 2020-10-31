"""
Implements multi-channel neural message masking.
"""
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU
from torch import Tensor
from torch.nn import ModuleList


class NMMConv(MessagePassing):
    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        """
        Neural message masking (NMM) layer. output of multiple instances of this
        will produce multi-channel output. 
        """
        super(NMMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(Linear(num_edge_attr, 8), 
                                           LeakyReLU(), 
                                           Linear(8, 1),
                                           ELU())
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters() 

    def forward(self, x, edge_index, edge_attr, size = None):
        if isinstance(x, Tensor):
            x = (x, x) # x: OptPairTensor

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


class MCNMMConv(torch.nn.Module):
    """
    Multi-channel neural message masking module.
    """
    def __init__(self, in_dim, out_dim, num_channels=1, num_edge_attr=1, train_eps=True, eps=0):
        super(MCNMMConv, self).__init__()
        self.nn = Sequential(Linear(in_dim * num_channels, out_dim), LeakyReLU(), Linear(out_dim, out_dim))
        self.NMMs = ModuleList()
        
        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NMMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
        # concatenate output of each channel
        x = troch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x

        

