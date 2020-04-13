import torch as th
import torch.nn as nn    
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import numpy.matlib as mtlb
from .utils import *

def erosion2d(phi, strel=3):
    if strel%2==0:
        padding = [ strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2) ] 
        tmp = -th.nn.functional.max_pool2d(-phi, strel, stride=1)
        return th.nn.ZeroPad2d(padding)(tmp)
    else:
        return -th.nn.functional.max_pool2d(-phi, strel, padding=strel//2, stride=1)

def dilatation2d(phi, strel=3):
    if strel%2==0:
        padding = [ strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2) ] 
        tmp = th.nn.functional.max_pool2d(phi, strel, stride=1)
        return th.nn.ZeroPad2d(padding)(tmp)
    else:
        return th.nn.functional.max_pool2d(phi, strel, padding=strel//2, stride=1)

def closing2d(phi, strel=3):
    return erosion2d(dilatation2d(phi, strel=strel), strel=strel)

def opening2d(phi, strel=3):
    return dilatation2d(erosion2d(phi, strel=3), strel=strel)

def grad2d(phi, strel=3):
    return  dilatation2d(phi, strel=strel) - erosion2d(phi, strel=strel)

def erosion3d(phi, strel=3):
    if strel%2==0:
        padding = [ strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2) ] 
        tmp = -th.nn.functional.max_pool3d(-phi, strel, stride=1)
        return th.nn.ZeroPad3d(padding)(tmp)
    else:
        return -th.nn.functional.max_pool3d(-phi, strel, padding=strel//2, stride=1)

def dilatation3d(phi, strel=3):
    if strel%2==0:
        padding = [ strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2), strel//2, strel//2 - (1-strel%2) ] 
        tmp = th.nn.functional.max_pool3d(phi, strel, stride=1)
        return th.nn.ZeroPad3d(padding)(tmp)
    else:
        return th.nn.functional.max_pool3d(phi, strel, padding=strel//2, stride=1)

def closing3d(phi, strel=3):
    return erosion3d(dilatation3d(phi, strel=strel), strel=strel)

def opening3d(phi, strel=3):
    return dilatation3d(erosion3d(phi, strel=3), strel=strel)

def grad3d(phi, strel=3):
    return  dilatation3d(phi, strel=strel) - erosion3d(phi, strel=strel)
    
class Erosion2d(nn.Module):
    def __init__(self, strel=3):
        super(Erosion2d, self).__init__()
        self.strel=strel
        
    def forward(self, x):
        x = erosion2d(x, strel=self.strel)
        return x
    
class Dilatation2d(nn.Module):
    def __init__(self, strel=3):
        super(Dilatation2d, self).__init__()
        self.strel=strel
        
    def forward(self, x):
        x = dilatation2d(x, strel=self.strel)
        return x
    
class Closing2d(nn.Module):
    def __init__(self, strel=3):
        super(Closing2d, self).__init__()
        self.strel=strel
        
    def forward(self, x):
        x = closing2d(x, strel=self.strel)
        return x
    
class Opening2d(nn.Module):
    def __init__(self, strel=3):
        super(Opening2d, self).__init__()
        self.strel=strel
        
    def forward(self, x):
        x = opening2d(x, strel=self.strel)
        return x
    
class MorphoGrad2d(nn.Module):
    def __init__(self, strel=3):
        super(MorphoGrad2d, self).__init__()
        self.strel = strel
        
    def forward(self, x):
        x = grad2d(x, strel=self.strel)
        return x
    
class Skeleton2d(th.nn.Module):
    def __init__(self, n=6):
        super(Skeleton2d, self).__init__()
        self.n = n        
        
    def forward(self,x):
        e0 = x
        sk = 0
        for i in range(self.n):
            ee0 = erosion2d(e0)
            o0 = dilatation2d(ee0)
            sk0 = e0 - o0
            sk += sk0*2.
            if i == self.n-1:
                return closing2d(sk + e0)
            e0 = ee0/2.
    
class Skeleton3d(th.nn.Module):
    def __init__(self, n=6):
        super(Skeleton3d, self).__init__()
        self.n = n        
        
    def forward(self,x):
        e0 = x
        sk = 0
        for i in range(self.n):
            ee0 = erosion3d(e0)
            o0 = dilatation3d(ee0)
            sk0 = e0 - o0
            sk += sk0*2.
            if i == self.n-1:
                return closing3d(sk + e0)
            e0 = ee0/2.
