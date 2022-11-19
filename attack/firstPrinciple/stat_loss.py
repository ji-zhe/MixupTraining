import os
import numpy as np
import torch
import torchvision
from models import *
import json
from scipy.stats import norm

filepath = './jsonfile/mix_outLoss.json'

def jload(filepath):
    f = open(filepath)
    tmp = json.load(f)
    f.close()
    return tmp

def jdump(obj, filepath):
    f = open(filepath,'w')
    json.dump(obj,f)
    f.close()

datasetIdx = jload('./jsonfile/idx.json')
losslist = jload(filepath)
# data_pos = torch.utils.data.Subset(datasetT, datasetIdx[0])
# data_neg = torch.utils.data.Subset(datasetT, np.setdiff1d(np.arange(10000),datasetIdx[0]))
resList = [] # (base ratio, mix ratio, membership)

for i in range(10000):
    itemLoss = losslist[i]

    stat = norm.fit(itemLoss)

    resList.append(stat)

jdump(resList, filepath[:-5]+'_mu_sigma.json')


