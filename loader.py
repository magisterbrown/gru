import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class EnergyDataset(Dataset):
    def __init__(self,df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,idx):
        row = self.df.iloc[idx,:] 
        row = row.to_numpy()
        return row[:-1],row[1:]
