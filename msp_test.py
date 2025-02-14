import torch
print('msp device:',torch.backends.mps.is_available())
print('cuda device:',torch.cuda.is_available())