import unittest
import pandas as pd
from loader import EnergyDataset

class Test(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("data/processed/train.csv",index_col=0)
        self.ds = EnergyDataset(self.df)
    
    def load(self):
        x,y = self.ds[0]

        self.assertEquals(x.size,90)
        self.assertEquals(y.size,90)
if __name__ == '__main__':
    unittest.main()

