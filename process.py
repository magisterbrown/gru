import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import pickle

class Stats():
    def __init__(self,mean: float,std: float):
        self.mean=mean
        self.std=std

train_size = 142000
batch_size = 90+1

files = list()

dirprocessed = "data/processed"
direc = "data/original"
f="PJME_hourly.csv"
fl = pd.read_csv(f'{direc}/{f}')
seq = fl.iloc[:,[1]].to_numpy().squeeze(axis=1)

ts = (int(train_size/batch_size)+1)*batch_size
train_x=seq[:ts]
batch_x= np.reshape(train_x, (-1,batch_size))
mean = train_x.mean()
std = train_x.std()
stats = Stats(mean,std)
norm = lambda x:(x-mean)/std
tdf = pd.DataFrame(norm(batch_x))

known = pd.Series(norm(train_x))
unkonwn = pd.Series(norm(seq[ts:]))
with open(f'{dirprocessed}/stats.pkl', 'wb') as out:
    pickle.dump(stats,out,pickle.HIGHEST_PROTOCOL)

tdf.to_csv(f'{dirprocessed}/train.csv')

known.to_csv(f'{dirprocessed}/test_x.csv')
unkonwn.to_csv(f'{dirprocessed}/test_y.csv')

