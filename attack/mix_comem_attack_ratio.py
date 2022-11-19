'''
execute after mix_comem_attack.py
'''

import torch
import torchvision
from torchvision import transforms
from scipy.stats import norm
from utils import *
import numpy as np 
import os
from matplotlib import pyplot as plt

# modelID = 0
# ref_mode = 'mix'
for modelID in [0]:
    for ref_mode in ['mix','base']:
        print(f'{ref_mode}Ref_modelID{modelID}')
        pos_scores = torch.load(f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_pos_scores.pt') #shape: (512,5000)
        neg_scores = torch.load(f'./ptfile/mixData_{ref_mode}Ref_modelID{modelID}_neg_scores.pt')
        mask = jload('./jsonfile/mask.json')
        datasetIdx = jload('./jsonfile/idx.json')

        mask = np.array(mask, dtype=bool).T
        pos_scores_min = np.minimum(pos_scores[0.2], pos_scores[0.5])
        pos_scores_min = np.minimum(pos_scores_min, pos_scores[0.8])
        pos_scores_avg = (pos_scores[0.2] + pos_scores[0.5] + pos_scores[0.8]) / 3
        neg_scores_min = np.minimum(neg_scores[0.2], neg_scores[0.5])
        neg_scores_min = np.minimum(neg_scores_min, neg_scores[0.8])
        neg_scores_avg = (neg_scores[0.2] + neg_scores[0.5] + neg_scores[0.8]) / 3


        trans_crop = transforms.CenterCrop(128)
        trnas_resize = transforms.Resize(64)
        trans_tensor = transforms.ToTensor()
        trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])

        datasetT = torchvision.datasets.CelebA('../dataset/', split= 'test', target_type= 'attr', transform = trans, target_transform = None, download = False)
        data_pos = torch.utils.data.Subset(datasetT, datasetIdx[modelID])
        data_neg = torch.utils.data.Subset(datasetT, np.setdiff1d(np.arange(10000),datasetIdx[modelID]))
        ratioList = [] # (base ratio, mix ratio, membership)
        bs = 100
        pos_loader = torch.utils.data.DataLoader(dataset=data_pos, batch_size=bs)
        neg_loader = torch.utils.data.DataLoader(dataset=data_neg, batch_size=bs)
        pos_loader2 = torch.utils.data.DataLoader(dataset=data_pos, batch_size=bs, shuffle=True)
        neg_loader2 = torch.utils.data.DataLoader(dataset=data_neg, batch_size=bs, shuffle=True)

        ratioList= []

        torch.manual_seed(0)
        for bid, ((img, label, iid), (img2, label2, iid2)) in enumerate(zip(pos_loader, pos_loader2)):
            # print(iid2); break
            for imgid in range(bs):
                global_imgid = bid*bs + imgid
                obs_loss = pos_scores_avg[modelID,global_imgid]
                in_model_mask = np.logical_and(mask[iid[imgid]], mask[iid2[imgid]])
                in_model_mask[modelID] = False
                out_model_mask = np.logical_and(np.logical_not(mask[iid[imgid]]) , np.logical_not(mask[iid2[imgid]]))
                out_model_mask[modelID] = False
                ref_in_loss = pos_scores_avg[in_model_mask[:128],global_imgid]
                ref_out_loss = pos_scores_avg[out_model_mask[:128],global_imgid]
                in_prob = norm.pdf(obs_loss,*norm.fit(ref_in_loss))
                out_prob = norm.pdf(obs_loss,*norm.fit(ref_out_loss))+1e-8
                if bid == 0:
                    plt.clf()
                    plt.hist(ref_in_loss, bins=25, alpha=0.5, label=f'in_model_{in_model_mask.sum()}', density=True)
                    plt.hist(ref_out_loss, bins=25, alpha=0.5, label=f'out_model_{out_model_mask.sum()}', density=True)
                    plt.legend()
                    plt.savefig(f'./fig/{modelID}_{global_imgid}_in_out.png') 
                ratioList.append((in_prob/out_prob, 1))
            # if bid > 3:
            #     exit()

        torch.manual_seed(0)
        for bid, ((img, label, iid), (img2, label2, iid2)) in enumerate(zip(neg_loader, neg_loader2)):
            # print(iid2); break
            for imgid in range(bs):
                global_imgid = bid*bs + imgid
                obs_loss = neg_scores_avg[modelID,global_imgid]
                in_model_mask = np.logical_and(mask[iid[imgid]], mask[iid2[imgid]])
                out_model_mask = np.logical_and(np.logical_not(mask[iid[imgid]]) , np.logical_not(mask[iid2[imgid]]))
                ref_in_loss = neg_scores_avg[in_model_mask[:128],global_imgid]
                ref_out_loss = neg_scores_avg[out_model_mask[:128],global_imgid]
                in_prob = norm.pdf(obs_loss,*norm.fit(ref_in_loss))
                out_prob = norm.pdf(obs_loss,*norm.fit(ref_out_loss))+1e-8
                ratioList.append((in_prob/out_prob, 0))
        os.makedirs(f'jsonfile/atk_mix_comem_{ref_mode}Ref/{modelID}', exist_ok=True)
        jdump(ratioList, f'jsonfile/atk_mix_comem_{ref_mode}Ref/{modelID}/atk_ratio_mix_avg.json')