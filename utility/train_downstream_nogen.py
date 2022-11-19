import torch
import numpy as np
from torch import optim
import os
import torchvision.utils as vutils
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import torchvision

BATCH_SIZE = 256
NZ = 10
max_iter = 10000
MIX_LABEL = False
n_class = 10
db_path = '../dataset'
gender_idx = 20
target_idx=gender_idx
bs = 64
# Data loaders
trans_t = transforms.ToTensor()
trans_c = transforms.CenterCrop(128)
trans_r = transforms.Resize((64,64))
trans = transforms.Compose([trans_c,trans_r, trans_t])
dataset = torchvision.datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
torch.manual_seed(0)
length = len(dataset)
dataset, _ = torch.utils.data.random_split(dataset, [4096, length-4096])

data_loader = torch.utils.data.DataLoader(dataset, bs)

# netC = Classifier().cuda()
netC = torchvision.models.resnet18(pretrained = True)
netC.fc = torch.nn.Linear(512,1)
netC = netC.cuda()
opt_c = torch.optim.Adam(netC.parameters(), lr=1e-2)
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for e in range(15):
    for img, label in data_loader:
        opt_c.zero_grad()
        img = img.to(device)
        pred = netC(img).squeeze()
        label_gpu = label[:,target_idx].to(device).type(torch.float)
        loss = loss_fn(pred, label_gpu)
        loss.backward()
        opt_c.step()
        print(loss.item())
    torch.save(netC.state_dict(), f'./models/netC_model/netC_realimg_e{e}.pkl')

