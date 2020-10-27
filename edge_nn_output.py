"""
Load the classifier model's edge_transformer neural network, and generate the 
edge weights it learned during training.
"""
import torch
from model import MoNet


if __name__ == "__main__":
    # hardware to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu

    # model path
    model_path = '../trained_models/trained_classifier_model_25.pt'

    # load the model
    model = MoNet(num_classes=14, num_features=11, dim=96, 
                  train_eps=True, num_edge_attr=1, which_model='jk', num_layers=6, deg=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    #print(model.embedding_net.conv0.edge_transformer)
    
    # artifical input
    input = torch.tensor([0, 1, 1.5, 2]).view(-1, 1)

    for i in range(6):
        # get edge nn 
        nn = eval('model.embedding_net.conv{}.edge_transformer'.format(i))

        # print output 
        out = nn(input)
        print(out.detach())