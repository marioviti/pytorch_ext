import torch as th
import torch.nn as nn    
import numpy as np

class Augmentation(nn.Module):
    
    def __init__(self):
        super(Augmentation, self).__init__()
    
    def forward(self,x):
        if not self.training: return x
        num_dims = len(x.shape)
        if np.random.rand() < 0.25:
            std = x.std()
            eps = ((np.random.rand()-1)*2 )*0.001
            x += std*eps
        if np.random.rand() < 0.25:
            dim0 = np.random.randint(2,num_dims)
            dim1 = np.random.randint(2,num_dims)
            #while dim0 == dim1 : dim1 = np.random.randint(2,num_dims)
            x = th.transpose(x, dim0, dim1)
        if np.random.rand() < 0.25:
            dim0 = np.random.randint(2,num_dims+1)
            dim1 = np.random.randint(2,num_dims+1)
            #while dim0 >= dim1 : dim1 = np.random.randint(2,num_dims+1)
            x = th.flip(x, list(range(2,5)))
        return x
            