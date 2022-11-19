from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

from models import GanGenerator as Generator
from models import GanDiscriminator as Discriminator
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
# parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--comem', type=int, default=5,help='comember strength')
parser.add_argument("--disc_num", type=int, default=2, help="num of pargan disc")
parser.add_argument("--data_num", type=int, default=4096, help="size of training dataset")
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of steps to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netBBG', type=str, required=True, help="path to netBBG (to attack)")
# parser.add_argument('--netWBG', default='', help="path to netWBG (to continue training)")
# parser.add_argument('--netBBD', type=str, required=True, help="path to netBBD (to attack)")
# parser.add_argument('--netWBD', default='', help="path to netWBD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except OSError:
    pass
print(opt)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Data
print('==> Preparing data..')

trans_n = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
trans_crop = transforms.CenterCrop(128)
trnas_resize = transforms.Resize(64)
trans_tensor = transforms.ToTensor()
trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])
# trans = transforms.Compose([trnas_resize, trans_tensor])
dataset = torchvision.datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
torch.manual_seed(0)
pos_data, rest_data = torch.utils.data.random_split(dataset, [opt.data_num, len(dataset)-opt.data_num])
neg_data, _ = torch.utils.data.random_split(rest_data, [opt.data_num, len(rest_data)-opt.data_num])

pos_loader = torch.utils.data.DataLoader(pos_data, batch_size=opt.batchSize, shuffle=True, drop_last = True)
neg_loader = torch.utils.data.DataLoader(neg_data, batch_size=opt.batchSize, shuffle=True, drop_last = True)

# device = torch.device("cuda:0" if opt.cuda else "cpu")
device = torch.device('cuda:0')
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
gender_idx = 20
target_idx=gender_idx
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netBBG = Generator(nz,y_dim=2).to(device)
# netBBG.load_state_dict(torch.load(opt.netBBG))
discList = []
for i in range(opt.disc_num):
    netBBD = Discriminator(nc, y_dim=2).to(device)
    discList.append(netBBD)

result_dir = f'./comem_logan_result/zdim{opt.nz}/datasetSize{opt.data_num}/'
os.makedirs(result_dir, exist_ok=True)

netBBG.eval()
netBBD.eval()
# netBBG_path="./GAN_model/params/G_{}.pkl"
# netBBD_path="./GAN_model/params/D_{}.pkl"
aucprclist = []
aucroclist = []
epochlist = []
##### White-box attack ####
# Assumes we have direct access to BBD
# for e in range(0,460,10):
# epochTestList=list(range(500,5000,500))
# epochRest = list(range(5000,30001,5000))
epochTestList = list(range(500, 30001,500))
# epochTestList.extend(epochRest)


netBBG_path=f"./models/zdim{opt.nz}/datasetSize{opt.data_num}/PARGAN_model/discNum{opt.disc_num}/params/G_{{}}.pkl"
netBBD_path=f"./models/zdim{opt.nz}/datasetSize{opt.data_num}/PARGAN_model/discNum{opt.disc_num}/params/D{{}}_{{}}.pkl"

aucroclist_pargan_sumStrategy = []
aucroclist_pargan_maxStrategy = []


for e in epochTestList:
    # epochlist2.append(e)
    wb_predictions = []
    wb_prob_sum = []
    wb_prob_max = []
    wb_label = []
    netBBG.load_state_dict(torch.load(netBBG_path.format(e)))
    for Did in range(opt.disc_num):
        discList[Did].load_state_dict(torch.load(netBBD_path.format(Did,e)))

    # loop over training data
    for i, data in enumerate(pos_loader, 0):
        real_cpu = data[0].to(device)
        real_cond = data[1][:,target_idx].to(device)
        real_cond = torch.nn.functional.one_hot(real_cond, 2)
        max_output = discList[0](real_cpu,real_cond)
        sum_output = 0
        for Did in range(opt.disc_num):
            single = discList[Did](real_cpu,real_cond)
            sum_output += single
            max_output = torch.max(max_output, single)
        # output = netBBD(real_cpu,real_cond)
        if opt.comem > 1:
            sum_outputs = torch.split(sum_output, opt.comem) 
            for bundle in sum_outputs:
                bundle.fill_(bundle.mean())
            sum_output = torch.cat(sum_outputs)

            max_outputs = torch.split(max_output, opt.comem) 
            for bundle in max_outputs:
                bundle.fill_(bundle.mean())
            max_output = torch.cat(max_outputs)
        wb_prob_sum.extend(sum_output.cpu())
        wb_prob_max.extend(max_output.cpu())
        wb_label.extend([1 for i in range(len(max_output))])

    # loop over test data
    for i, data in enumerate(neg_loader, 0):
        real_cpu = data[0].to(device)
        real_cond = data[1][:,target_idx].to(device)
        real_cond = torch.nn.functional.one_hot(real_cond, 2)
        max_output = discList[0](real_cpu,real_cond)
        sum_output = 0
        for Did in range(opt.disc_num):
            single = discList[Did](real_cpu,real_cond)
            sum_output += single
            max_output = torch.max(max_output, single)
        # output = netBBD(real_cpu,real_cond)
        if opt.comem > 1:
            sum_outputs = torch.split(sum_output, opt.comem) 
            for bundle in sum_outputs:
                bundle.fill_(bundle.mean())
            sum_output = torch.cat(sum_outputs)
            
            max_outputs = torch.split(max_output, opt.comem) 
            for bundle in max_outputs:
                bundle.fill_(bundle.mean())
            max_output = torch.cat(max_outputs)
        wb_prob_sum.extend(sum_output.cpu())
        wb_prob_max.extend(max_output.cpu())
        wb_label.extend([0 for i in range(len(max_output))])
    # import pdb;pdb.set_trace()
    # wb_predictions = [x[1] for x in sorted(wb_predictions, reverse=True)[:len(pos_data)]]
    # wb_accuracy = wb_predictions.count('train')/float(len(pos_data))
    print()
    print("Iteration:", e)
    # print("baseline (random guess) accuracy: {:.3f}".format(len(pos_data)/float(len(pos_data)+len(neg_data))))
    # print("white-box attack accuracy: {:.3f}".format(wb_accuracy))


    y_test = wb_label
    y_score = wb_prob_max
    print("max strategy")
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve
    from matplotlib import pyplot as plt 
    import numpy as np
    AUCROC = roc_auc_score(y_test, y_score)
    aucroclist_pargan_maxStrategy.append(AUCROC)
    fpr, tpr, thres = roc_curve(y_test, y_score)
    print("AUCROC:", AUCROC)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC curve")
    # plt.plot(fpr, tpr)
    # plt.savefig(f"./fig/ROC_curve_{suffix}.png")
    plt.clf()
    prec, rec, thr = precision_recall_curve(y_test, y_score)
    tmp = np.vstack((rec,prec))
    tmp = np.transpose(tmp)
    tmp = tmp[np.argsort(tmp[:,0])]
    aucprc = auc(tmp[:,0], tmp[:,1])
    print("AUCPRC:", aucprc)
    # aucroclist_pargan.append(aucprc)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("PR curve")
    # plt.plot(rec, prec)
    # plt.savefig(f"./fig/PR_curve_{suffix}.png")

    y_test = wb_label
    y_score = wb_prob_sum
    print("sum strategy")
    AUCROC = roc_auc_score(y_test, y_score)
    aucroclist_pargan_sumStrategy.append(AUCROC)
    fpr, tpr, thres = roc_curve(y_test, y_score)
    print("AUCROC:", AUCROC)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC curve")
    # plt.plot(fpr, tpr)
    # plt.savefig(f"./fig/ROC_curve_{suffix}.png")
    plt.clf()
    prec, rec, thr = precision_recall_curve(y_test, y_score)
    tmp = np.vstack((rec,prec))
    tmp = np.transpose(tmp)
    tmp = tmp[np.argsort(tmp[:,0])]
    aucprc = auc(tmp[:,0], tmp[:,1])
    print("AUCPRC:", aucprc)

torch.save(aucroclist_pargan_maxStrategy, os.path.join(result_dir,f'pargan_discNum{opt.disc_num}_comem{opt.comem}_max.pkl'))
torch.save(aucroclist_pargan_sumStrategy, os.path.join(result_dir,f'pargan_discNum{opt.disc_num}_comem{opt.comem}_sum.pkl'))

plt.clf()
plt.ylim(0.4,1)
# plt.plot(epochTestList, aucroclist, label='base')
# plt.plot(epochTestList, aucroclist2, label='mixup')
# plt.plot(epochTestList, aucroclist3, label='1G2D-D1')
# plt.plot(epochTestList, aucroclist4, label='1G2D-D2')
plt.plot(epochTestList, aucroclist_pargan_maxStrategy, label='PARGAN max Strategy')
plt.plot(epochTestList, aucroclist_pargan_sumStrategy, label='PARGAN sum Strategy')
plt.xlabel('Num of Iteration')
plt.ylabel('Attack AUCROC')
# plt.plot(epochlist, aucprclist, label='aucprc')
plt.legend()
plt.savefig(f"pargan_{opt.batchSize}_z{opt.nz}_disc{opt.disc_num}_comem{opt.comem}.png")
# plt.savefig(f"auc_iteration_logan_comem_{opt.batchSize}_z{opt.nz}.png")

exit()

##### Black-box attack ####
# Trains another GAN on the output of the black-box
# Then launches whitebox attack with trained Discriminator 
