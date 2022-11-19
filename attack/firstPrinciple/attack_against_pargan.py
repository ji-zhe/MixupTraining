import os
import numpy as np
import torch
import torchvision
from models import *
import json
from scipy.stats import norm
from torchvision import transforms

latent_dim = 256
n_class = 2
modelID = 0
target_idx = 20
mode = 'pargan'
for modelID in range(1):
    target_G = f'./models/{mode}/{modelID}/params/G_799.pkl'
    target_D = f'./models/{mode}/{modelID}/params/D_{{}}_799.pkl'
    # target_G = f'../wgan_torch_celebA/models/zdim256/datasetSize4096/PARGAN_model/discNum4/params/G_30000.pkl'
    # target_D = f'../wgan_torch_celebA/models/zdim256/datasetSize4096/PARGAN_model/discNum4/params/D{{}}_30000.pkl'

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
    discList = []
    for i in range(4):
        discriminator = GanDiscriminator(y_dim=n_class)
        discriminator = nn.DataParallel(discriminator).cuda()
        discriminator.module.load_state_dict(torch.load(target_D.format(i)))
        discriminator.eval()
        discList.append(discriminator)

    generator.eval()
    # discriminator.eval()

    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)
    # data_pos = torch.utils.data.Subset(datasetT, datasetIdx[0])
    # data_neg = torch.utils.data.Subset(datasetT, np.setdiff1d(np.arange(10000),datasetIdx[0]))
    ratioList = [] # (base ratio, mix ratio, membership)

    for i, (img, label, _) in enumerate(datasetT):
        assert i == _
        if i == 10000:
            break
        label = label[target_idx]
        bin = base_inLoss[i]
        bout = base_outLoss[i]
        m_in = mix_inLoss[i]
        mout = mix_outLoss[i]

        y = torch.ones((64,),dtype=int).cuda() * label 
        y = torch.nn.functional.one_hot(y, n_class).cuda()
        z = torch.randn((64,latent_dim)).cuda()
        fake = generator(z,y)
        # import pdb;pdb.set_trace()
        min_diff = 1e10
        max_diff = -1e10
        for did in range(4):
            discriminator = discList[did]
            real_validity = discriminator(img.unsqueeze(0), y[0:1])
            fake_validity = discriminator(fake, y).mean()
            diff = fake_validity - real_validity
            if diff < min_diff:
                min_diff = diff
            if diff > max_diff:
                max_diff = diff

        # import pdb; pdb.set_trace()
        obf = min_diff.item()
        
        bin_prob = norm.pdf(obf, *norm.fit(bin))
        bout_prob = norm.pdf(obf, *norm.fit(bout))
        min_prob =  norm.pdf(obf, *norm.fit(m_in))
        mout_prob =  norm.pdf(obf, *norm.fit(mout))
        ratioList.append((bin_prob/bout_prob, min_prob/mout_prob, int((i in memLabel))))

    os.makedirs(f'./jsonfile/atk_ratio/{modelID}/', exist_ok=True)
    jdump(ratioList, f'./jsonfile/atk_ratio/{modelID}/atk_ratio_{mode}.json')


