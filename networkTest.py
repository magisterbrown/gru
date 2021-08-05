import unittest
import pandas as pd
from network import RNN
from torch.utils.data import DataLoader
from loader import EnergyDataset

class Test(unittest.TestCase):
    rnn = RNN()
    def setUp(self):
        
        df = pd.read_csv("data/processed/train.csv",index_col=0)
        ds = EnergyDataset(df)
        self.dl = DataLoader(ds,batch_size=16,shuffle=True)

    def runt(self):
        batch = next(iter(self.dl)) 
        out = self.rnn(batch[0].float())        
        self.assertEqual(out.shape,batch[1].shape)
if __name__ == '__main__':
    unittest.main()
