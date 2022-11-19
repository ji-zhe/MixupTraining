import torch
import torchvision
from torchvision import datasets, transforms
from models import *
import sklearn.metrics
import numpy as np
import os
# netC_path = './netC_model/resnet18/base/netC_base_singleLabel_e{}.pkl'
netC_path = './netC_model/relaxLoss_squareOnly/2/netC_singleLabel_e{}.pkl'
print("netC:", netC_path)
BATCH_SIZE=256

os.environ['CUDA_VISIBLE_DEVICES']='2'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset = torchvision.datasets.CIFAR10('../dataset/cifar10', train=True, transform = transform_test, target_transform = None, download = False)

testLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                          drop_last=True)


# netC = Classifier()
netC = torchvision.models.resnet18(pretrained = True)
netC.fc = torch.nn.Linear(512,10)
netC = netC.cuda()
# netC.load_state_dict(torch.load(netC_path))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netC.to(device)
netC.eval()

with torch.no_grad():
    # import pdb;pdb.set_trace()
    for e in range(0,3001,200):
        predlist = list()
        labellist = list()
        netC.load_state_dict(torch.load(netC_path.format(e)))
        for img, label, _ in testLoader:
            img = img.to(device)
            pred = torch.softmax(netC(img),dim=1)
            # pred = torch.sigmoid(netC(img))
            predlist.append(pred.cpu())
            labellist.append(label)
        y_true = torch.cat(labellist).numpy()
        # y_score = torch.vstack(predlist).numpy()
        y_score = torch.vstack(predlist).numpy()
        # import pdb;pdb.set_trace()
        auc = sklearn.metrics.roc_auc_score(y_true, y_score, multi_class='ovr') # 
        y_pred = np.argmax(y_score, axis=1)
        # y_pred = (y_score>0.5).astype(int)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        print("Epoch:", e, end='\t')
        print("acc:",acc)
        # print("auc:",auc)
        # exit()

