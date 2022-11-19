import torch
from models import GanGenerator
import numpy as np
import torchvision
import argparse



parser = argparse.ArgumentParser()
# parser.add_argument("--mixup", action='store_true', help="mixup or not in GAN training")
parser.add_argument("--mixLabel", action='store_true', help="mix label or single label")

opt = parser.parse_args()
print(opt)

BATCH_SIZE = 256
NZ = 256
max_iter = 1200
MIX_LABEL = opt.mixLabel
n_class = 2
# mixup = opt.mixup

# netC = Classifier().cuda()
netC = torchvision.models.resnet18(pretrained = True)
netC.fc = torch.nn.Linear(512,1)
netC = netC.cuda()
netG = GanGenerator(z_dim=NZ, y_dim=2)
# netG.load_state_dict(torch.load("./models/zdim256/datasetSize4096/{}GAN_model/params/G_4000.pkl".format('mix' if mixup else '')))
netG.load_state_dict(torch.load("./models/zdim256/datasetSize4096/1g2d_model/params/G_10000.pkl"))
netG = netG.cuda()
netG.eval()

opt_c = torch.optim.Adam(netC.parameters(), lr=1e-2)
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for _ in range(max_iter):
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
    else: 
        img = netG(noise, rand_cond_onehot)
    pred = netC(img).squeeze()
    if MIX_LABEL:
        rand_cond = rand_cond.type(torch.float)
        rand_cond2 = rand_cond2.type(torch.float)
        loss = loss_fn(pred, rand_cond) * lam + loss_fn(pred, rand_cond2) * (1-lam)
    else:
        # import pdb;pdb.set_trace()
        rand_cond = rand_cond.type(torch.float)
        loss = loss_fn(pred, rand_cond)
    loss.backward()
    opt_c.step()
    print(loss.item())
torch.save(netC.state_dict(), './models/zdim256/datasetSize4096/netC_model/1g2d/netC_10k_Giter.pkl')

