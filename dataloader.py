"""
Dataloader for the Siamese graph neural network. 
"""
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader


class PairData(Data):
    """
    Paired data type. Each object has 2 graphs.
    """
    def __init__(self,x_a, edge_index_a, edge_attr_a, x_b, edge_index_b, edge_attr_b):
        super(PairData, self).__init__()
        self.x_a = x_a
        self.edge_index_a = edge_index_a
        self.edge_attr_a = edge_attr_a

        self.x_b = x_b
        self.edge_index_b = edge_index_b        
        self.edge_attr_b = edge_attr_b

    def __inc__(self, key, value):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        else:
            return super(PairData, self).__inc__(key, value)
    

if __name__=="__main__":
    edge_index_a = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    x_a = torch.randn(5, 8)  # 5 nodes.
    edge_attr_a = torch.randn(4, 2)

    edge_index_b = torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
    ])
    x_b = torch.randn(4, 8)  # 4 nodes.
    edge_attr_b = torch.randn(3, 2)

    y = torch.tensor([1])

    data = PairData(x_a, edge_index_a, edge_attr_a, x_b, edge_index_b, edge_attr_b)
    data_list = [data, data]
    loader = DataLoader(data_list, batch_size=2, follow_batch=['x_a', 'x_b'])
    batch_data = next(iter(loader))

    print(batch_data)
    print(batch_data.edge_index_a)
    print(batch_data.edge_index_b)
    print(batch_data.x_a_batch)
    print(batch_data.x_b_batch)