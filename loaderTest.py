import unittest
import pandas as pd
from loader import EnergyDataset

class Test(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("data/processed/train.csv",index_col=0)
        self.ds = EnergyDataset(self.df)
    
    def load(self):
        x,y = self.ds[0]
        print(x.shape)
        print(y.shape)

if __name__ == '__main__':
    unittest.main()

