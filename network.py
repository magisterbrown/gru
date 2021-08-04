import numpy as np
import torch.nn as nn
import torch
from gru import GRU

class RNN(nn.module):
    def __init__(self,hidden=256):
        super().__init__()
        self.gru = GRU(1,hidden)
        self.rl = nn.ReLU()
        self.l1 = nn.Linear(hidden,1)
        self.hidden = hidden

    def forward(self,x):
        out = torch.zeros(x.shape)
        bs = x.shape[0]
        hid = torch.zeros((bs,self.hidden))
        for k,v in enumerate(x):
            hid = self.gru(v,hid)
            fin = self.l1(self.rl(hid))
            out[k]=fin
        
        return out