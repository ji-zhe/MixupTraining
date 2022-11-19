import os
import numpy as np
import torch
import torchvision
from models import *
import json
from scipy.stats import norm
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

latent_dim = 256
n_class = 2
modelID = 0
for mode in ['relaxLoss_changeAllSign']:
    target_G = f'./models/{mode}/{modelID}/params/G_199.pkl'
    target_D = f'./models/{mode}/{modelID}/params/D_199.pkl'
    print(target_D)

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
    base_inLoss = jload('./jsonfile/base_inLoss.json')
    base_outLoss = jload('./jsonfile/base_outLoss.json')
    mix_inLoss = jload('./jsonfile/mix_inLoss.json')
    mix_outLoss = jload('./jsonfile/mix_outLoss.json')

    memLabel = datasetIdx[modelID]

    generator = GanGenerator(z_dim=latent_dim, y_dim=n_class)
    generator = nn.DataParallel(generator).cuda()
    generator.module.load_state_dict(torch.load(target_G))
    discriminator = GanDiscriminator(y_dim=n_class)
    discriminator = nn.DataParallel(discriminator).cuda()
    discriminator.module.load_state_dict(torch.load(target_D))

    generator.eval()
    discriminator.eval()

    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)
    # data_pos = torch.utils.data.Subset(datasetT, datasetIdx[0])
    # data_neg = torch.utils.data.Subset(datasetT, np.setdiff1d(np.arange(10000),datasetIdx[0]))
    scoreList = [] # (base ratio, mix ratio, membership)

    for i, (img, label,_) in enumerate(datasetT):

        y = torch.ones((64,),dtype=int).cuda() * label[20] 
        y = torch.nn.functional.one_hot(y, n_class).cuda()
        z = torch.randn((64,latent_dim)).cuda()
        fake = generator(z,y)
        real_validity = discriminator(img.unsqueeze(0), y[0:1])

        scoreList.append((real_validity.item(), int((i in memLabel))))

    os.makedirs(f'./jsonfile/logan/{modelID}/', exist_ok=True)
    jdump(scoreList, f'./jsonfile/logan/{modelID}/atkScore_{mode}.json')


