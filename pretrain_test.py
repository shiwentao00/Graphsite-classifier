import torch
from model import SiameseNet
from model import SelectiveSiameseNet
import copy


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')  # detect cpu or gpu

    features_to_use = ['x', 'y', 'z',  'r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
                       'binding_probability', 'sequence_entropy']
    num_features = len(features_to_use)

    model_0 = SiameseNet(num_features=num_features, dim=32, train_eps=True, num_edge_attr=1).to(device)
    model_1 = SelectiveSiameseNet(num_features=num_features, dim=32, train_eps=True, num_edge_attr=1).to(device)

    print(model_0.embedding_net.state_dict()['conv2.nn.2.weight'])
    print(model_1.embedding_net.state_dict()['conv2.nn.2.weight'])

    model_1.embedding_net.load_state_dict(copy.deepcopy(model_0.embedding_net.state_dict()))
    print(model_1.embedding_net.state_dict()['conv2.nn.2.weight'])