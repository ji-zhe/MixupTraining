import torch
import torchvision
from torchvision import datasets, transforms
from models import *
import sklearn.metrics
import numpy as np
netC_path = './models/zdim256/datasetSize4096/netC_model/mixup/netC_singleLabel_{}k_Giter.pkl'
BATCH_SIZE=256

trans_t = transforms.ToTensor()
trans_c = transforms.CenterCrop(128)
trans_r = transforms.Resize((64,64))
trans = transforms.Compose([trans_c, trans_r, trans_t])
dataset = datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
torch.manual_seed(0)
length = len(dataset)
_, testset = torch.utils.data.random_split(dataset, [4096, length-4096])
testLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                          drop_last=True)


# netC = Classifier()
netC = torchvision.models.resnet18(pretrained = True)
netC.fc = torch.nn.Linear(512,1)
netC = netC.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netC.to(device)
netC.eval()

for e in [1,2,3,4,5,10]:
    print("iter:", e,'k')
    netC.load_state_dict(torch.load(netC_path.format(e)))
    predlist = list()
    labellist = list()
    gender_idx = 20
    target_idx=gender_idx
    with torch.no_grad():
        # import pdb;pdb.set_trace()
        for img, label in testLoader:
            img = img.to(device)
            pred = torch.sigmoid(netC(img))
            predlist.append(pred.cpu())
            labellist.append(label[:,target_idx])
        y_true = torch.cat(labellist).numpy()
        # y_score = torch.vstack(predlist).numpy()
        y_score = torch.vstack(predlist).numpy()
        # import pdb;pdb.set_trace()
        auc = sklearn.metrics.roc_auc_score(y_true, y_score) # , multi_class='ovr'
        # y_pred = np.argmax(y_score, axis=1)
        y_pred = (y_score>0.5).astype(int)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        print("acc:",acc)
        print("auc:",auc)

