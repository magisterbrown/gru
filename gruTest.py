import unittest
import torch
from gru import GRU,InputGate
import pdb
import torch.nn as nn

class Test(unittest.TestCase):
    gru = GRU(5,10)
    inp = InputGate(5,10,nn.Sigmoid())

    def gateTest(self):
        inp = torch.rand(2,5) 
        hid = torch.rand(2,10)
        out = self.inp(inp,hid)
        self.assertEqual(list(out.shape),[2,10]) 

    def gruTest(self):
        inp = torch.rand(2,5) 
        hid = torch.rand(2,10)
        out = self.gru(inp,hid)
        self.assertEqual(list(out.shape),[2,10]) 


if __name__ == "__main__":
    unittest.main()

    
