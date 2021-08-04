import numpy as np
import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.reset = InputGate(inputs,hidden,nn.Sigmoid())
        self.update = InputGate(inputs,hidden,nn.Sigmoid())
        self.candidate = InputGate(inputs,hidden,nn.Tanh())        


    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        reswei = self.reset(x,memory)
        upwei = self.update(x,memory)
        
        mempart = memory*reswei
        canrem = self.candidate(x,mempart)

        savemem = memory*(1-upwei)
        newmem = canrem*upwei

        return savemem+newmem


class InputGate(nn.Module):
    def __init__(self, inputs: int, hidden: int, activation: nn.Module):
        super().__init__()

        self.memory = nn.Linear(hidden,hidden)
        self.input = nn.Linear(inputs,hidden)
        self.sig = activation

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        inp =self.input(x) 
        mem = self.memory(hidden)

        x = + inp+mem
        x = self.sig(x)
        return x
