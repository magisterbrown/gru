import numpy as np
import torch.nn as nn
import torch
from grun import GRU

class RNN(nn.Module):
    def __init__(self,hidden=256):
        super().__init__()
        self.gru = GRU(1,hidden)
        self.rl = nn.ReLU()
        self.l1 = nn.Linear(hidden,1)
        self.hidden = hidden

    def forward(self,x):
        out = torch.zeros(x.shape,dtype=x.dtype,device=x.device)
        bs = x.shape[0]
        hid = torch.zeros((bs,self.hidden),dtype=x.dtype,device=x.device)
        for i in range(x.shape[-1]):
            v = x[...,i][...,np.newaxis]
            hid = self.gru(v,hid)
            fin = self.l1(self.rl(hid))
            out[:,i]=fin.squeeze()
        
        return out
