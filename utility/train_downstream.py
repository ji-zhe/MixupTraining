import torch
from models import GanGenerator
import numpy as np
import torchvision
import argparse
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mixup", action='store_true', help="mixup or not in GAN training")
parser.add_argument("--mixLabel", action='store_true', help="mix label or single label")

opt = parser.parse_args()
print(opt)

BATCH_SIZE = 256
NZ = 256
max_iter = 3001
MIX_LABEL = opt.mixLabel
n_class = 2
mixup = opt.mixup
prtmode = 'base'
modelID = 2
# for prtmode in ['relaxLoss_squareOnly']:
for modelID in range(1):
    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    transform_train = trans= transforms.Compose([trans_crop, trnas_resize, trans_tensor])

    datasetT = torchvision.datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
    dataset, _ = torch.utils.data.random_split(datasetT, [10000, len(datasetT) - 10000])

    testLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                            drop_last=True)

    # netC = Classifier().cuda()
    netC = torchvision.models.resnet18(pretrained = True)
    netC.fc = torch.nn.Linear(512,10)
    netC = netC.cuda()
    netG = GanGenerator(z_dim=NZ, y_dim=n_class)
    netG.load_state_dict(torch.load(f"./models/{prtmode}/{modelID}/params/G_199.pkl"))
    # netG.load_state_dict(torch.load("./1g2d_GAN_model/params/G_190.pkl"))
    # netG.load_state_dict(torch.load("./{}GAN_model/params/G_190.pkl".format('mix' if mixup else '')))
    netG = netG.cuda()

    opt_c = torch.optim.Adam(netC.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for e in range(max_iter):
        noise = torch.randn(BATCH_SIZE, NZ, device=device)
        rand_cond = torch.randint(n_class, (BATCH_SIZE,), device=device)
        rand_cond_onehot = torch.nn.functional.one_hot(rand_cond, n_class).type(torch.float)
        lam = np.random.beta(1,1)
        opt_c.zero_grad()
        if MIX_LABEL:
            rand_cond2 = torch.randint(n_class, (BATCH_SIZE,), device=device)
            rand_cond2_onehot = torch.nn.functional.one_hot(rand_cond2, n_class).type(torch.float)
            rand_cond_mix = lam * rand_cond_onehot + (1-lam) * rand_cond2_onehot
            img = netG(noise, rand_cond_mix)
            # img = transform_train(img)
        else: 
            img = netG(noise, rand_cond_onehot)
            # img = transform_train(img)
        pred = netC(img).squeeze()
        if MIX_LABEL:
            # rand_cond = rand_cond.type(torch.float)
            # rand_cond2 = rand_cond2.type(torch.float)
            loss = loss_fn(pred, rand_cond) * lam + loss_fn(pred, rand_cond2) * (1-lam)
        else:
            # import pdb;pdb.set_trace()
            # rand_cond = rand_cond.type(torch.float)
            loss = loss_fn(pred, rand_cond)
        loss.backward()
        opt_c.step()
        # print(loss.item())
        if e % 200 == 0:
            with torch.no_grad():
        # import pdb;pdb.set_trace()
                netC.eval()
                predlist = list()
                labellist = list()
                for img, label, _ in testLoader:
                    img = img.to(device)
                    label = label[:,20]
                    pred = torch.softmax(netC(img),dim=1)
                    # pred = torch.sigmoid(netC(img))
                    predlist.append(pred.cpu())
                    labellist.append(label)
                y_true = torch.cat(labellist).numpy()
                # y_score = torch.vstack(predlist).numpy()
                y_score = torch.vstack(predlist).numpy()
                # import pdb;pdb.set_trace()
                # auc = sklearn.metrics.roc_auc_score(y_true, y_score, multi_class='ovr') # 
                y_pred = np.argmax(y_score, axis=1)
                # y_pred = (y_score>0.5).astype(int)
                acc = sklearn.metrics.accuracy_score(y_true, y_pred)
                print("Epoch:", e)
                print("acc:",acc)
                # print("auc:",auc)
                netC.train()
            os.makedirs(f'./netC_model/{prtmode}/{modelID}', exist_ok=True)
            torch.save(netC.state_dict(), f'./netC_model/{prtmode}/{modelID}/netC_{{}}_e{{}}.pkl'.format('mixLabel' if MIX_LABEL else 'singleLabel', e))

