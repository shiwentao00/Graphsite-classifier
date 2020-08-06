import torch 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
#print('device: ', device)
print('.........')
if torch.cuda.is_available():
    num_gpu = torch.cuda.device_count()
    print('number of gpus: ', num_gpu)