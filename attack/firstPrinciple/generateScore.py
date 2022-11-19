import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import grad

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import urllib3
from models import *
import json

gender_idx = 20
target_idx=gender_idx

if __name__ == "__main__":
    f = open('./jsonfile/idx.json','r')
    datasetIdx = json.load(f)
    f.close()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--n_samp", type=int, default=10, help="number of sampl")
    parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
    parser.add_argument("--n_class", type=int, default=2, help="number of classes")
    parser.add_argument("--mode", type=str, default='base', help="mode of training")
    opt = parser.parse_args()
    print(opt)
    lambda_gp = 10
    latent_dim = opt.latent_dim
    n_class = opt.n_class
    
    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)

    generator = GanGenerator(z_dim=latent_dim, y_dim=n_class)
    generator = nn.DataParallel(generator).cuda()
    generator.eval()
    discriminator = GanDiscriminator(y_dim=n_class)
    discriminator = nn.DataParallel(discriminator).cuda()
    discriminator.eval()
    torch.autograd.set_grad_enabled(False)

    mode = opt.mode 
    result_dir =  f'./atk_result/{mode}'
    os.makedirs(result_dir, exist_ok=True)
    bs = 64

    loader = torch.utils.data.DataLoader(datasetT, bs)
    scoreList = []

    for modelID in range(128):
        print(modelID)
        modelScore = []
        generator.module.load_state_dict(torch.load(f'./models/{mode}/{modelID}/params/G_199.pkl'))
        discriminator.module.load_state_dict(torch.load(f'./models/{mode}/{modelID}/params/D_199.pkl'))
        for imgs, labels, _ in loader:
            imgs = imgs.cuda()
            labels = labels[:, target_idx]
            z = torch.randn((imgs.shape[0],latent_dim)).cuda()
            y = torch.nn.functional.one_hot(labels, n_class).cuda()
            fake = generator(z,y)
            fake_validity = discriminator(fake, y)
            real_validity = discriminator(imgs, y)
            loss = -real_validity + fake_validity
            modelScore.append(loss)
        modelSumm = torch.cat(modelScore)
        scoreList.append(modelSumm)
    score = torch.vstack(scoreList).tolist()
    f = open(f'./jsonfile/{mode}_scoreAll.json','w')
    json.dump(score,f)
    f.close()