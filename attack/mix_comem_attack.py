import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from models import *
import json
from scipy.stats import norm

torch.autograd.set_grad_enabled(False)
latent_dim = 256
n_class = 2
modelID = None
mode = 'mix'
ref_mode = 'base'
print("modelID:", modelID, "\tRef_mode:", ref_mode, "\t prt_mode:", mode)


def jload(filepath):
    f = open(filepath)
    tmp = json.load(f)
    f.close()
    return tmp

def jdump(obj, filepath):
    f = open(filepath,'w')
    json.dump(obj,f)
    f.close()

for modelID in range(3):
    target_G = f'./models/{mode}/{modelID}/params/G_199.pkl'
    target_D = f'./models/{mode}/{modelID}/params/D_199.pkl'


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

    generator_ref = GanGenerator(z_dim=latent_dim, y_dim=n_class)
    generator_ref = nn.DataParallel(generator_ref).cuda()
    discriminator_ref = GanDiscriminator(y_dim=n_class)
    discriminator_ref = nn.DataParallel(discriminator_ref).cuda()

    generator_ref.eval()
    discriminator_ref.eval()

    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)
    # import pdb; pdb.set_trace()
    data_pos = torch.utils.data.Subset(datasetT, datasetIdx[modelID])
    data_neg = torch.utils.data.Subset(datasetT, np.setdiff1d(np.arange(10000),datasetIdx[modelID]))
    ratioList = [] # (base ratio, mix ratio, membership)
    bs = 100
    pos_loader = torch.utils.data.DataLoader(dataset=data_pos, batch_size=bs)
    neg_loader = torch.utils.data.DataLoader(dataset=data_neg, batch_size=bs)
    pos_loader2 = torch.utils.data.DataLoader(dataset=data_pos, batch_size=bs, shuffle=True)
    neg_loader2 = torch.utils.data.DataLoader(dataset=data_neg, batch_size=bs, shuffle=True)

    datsetIdx = np.array(datasetIdx)
    mask = jload('jsonfile/mask.json')
    mask = np.array(mask).T
    scoreList_pos = {0.2: list([] for _ in range(len(pos_loader))), 0.5: list([] for _ in range(len(pos_loader))), 0.8: list([] for _ in range(len(pos_loader)))}
    scoreList_neg = {0.2: list([] for _ in range(len(neg_loader))), 0.5: list([] for _ in range(len(neg_loader))), 0.8: list([] for _ in range(len(neg_loader)))}
    ratioList = []
    pos_loader2_idx = []
    neg_loader2_idx = []
    seed = 0
    for model_refID in range(128):
        print(f'{model_refID}/512')
        ref_G = f'./models/{ref_mode}/{model_refID}/params/G_199.pkl'
        ref_D = f'./models/{ref_mode}/{model_refID}/params/D_199.pkl'
        generator_ref.module.load_state_dict(torch.load(ref_G))
        discriminator_ref.module.load_state_dict(torch.load(ref_D))
        generator_ref.eval()
        discriminator_ref.eval()
        torch.manual_seed(seed)

        for i, ((img, label, iid), (img2, label2, iid2)) in enumerate(zip(pos_loader, pos_loader2)):
            if i == 0:
                pos_loader2_idx.append(iid2.numpy())
            img = img.cuda()
            img2 = img2.cuda()
            label = label[:,20]
            label2 = label2[:,20]
            label = torch.nn.functional.one_hot(label, num_classes=n_class).cuda()
            label2 = torch.nn.functional.one_hot(label2, num_classes=n_class).cuda()
            for lam in [0.2,0.5,0.8]:
                mix_img = lam * img + (1-lam) * img2
                mix_label = lam * label + (1-lam) * label2
                z = torch.randn(mix_img.shape[0], latent_dim).cuda()
                fake_img = generator_ref(z, mix_label)
                fake_validity = discriminator_ref(fake_img, mix_label)
                real_validity = discriminator_ref(mix_img, mix_label)
                scoreList_pos[lam][i].append((fake_validity-real_validity).detach().cpu().numpy())

        torch.manual_seed(seed)
        for i, ((img, label, iid), (img2, label2, iid2)) in enumerate(zip(neg_loader, neg_loader2)):
            if i == 0:
                neg_loader2_idx.append(iid2.numpy())
            img = img.cuda()
            img2 = img2.cuda()
            label = label[:,20]
            label2 = label2[:,20]
            label = torch.nn.functional.one_hot(label, num_classes=n_class).cuda()
            label2 = torch.nn.functional.one_hot(label2, num_classes=n_class).cuda()
            for lam in [0.2,0.5,0.8]:
                mix_img = lam * img + (1-lam) * img2
                mix_label = lam * label + (1-lam) * label2
                z = torch.randn(mix_img.shape[0], latent_dim).cuda()
                fake_img = generator_ref(z, mix_label)
                fake_validity = discriminator_ref(fake_img, mix_label)
                real_validity = discriminator_ref(mix_img, mix_label)
                scoreList_neg[lam][i].append((fake_validity-real_validity).detach().cpu().numpy()) # shape: (100,)
    torch.save(scoreList_pos,f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_pos_scores_raw.pt')
    for lam in [0.2,0.5,0.8]:
        for i in range(len(pos_loader)):
            scoreList_pos[lam][i] = np.vstack(scoreList_pos[lam][i]) # shape: (512, 100)
        scoreList_pos[lam] = np.concatenate(scoreList_pos[lam],axis=1)
    torch.save(scoreList_pos,f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_pos_scores.pt')

    torch.save(scoreList_neg,f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_neg_scores_raw.pt')
    for lam in [0.2,0.5,0.8]:
        for i in range(len(neg_loader)):
            scoreList_neg[lam][i] = np.vstack(scoreList_neg[lam][i])
        scoreList_neg[lam] = np.concatenate(scoreList_neg[lam],axis=1)
    torch.save(scoreList_neg,f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_neg_scores.pt')