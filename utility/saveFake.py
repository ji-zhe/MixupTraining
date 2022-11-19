import torch
import torchvision
import models
import os
from torchvision import transforms
NZ=256
YD=2
data_num = 4096
torch.set_grad_enabled(False)

# result_dir = 'real4096'
# os.makedirs(result_dir, exist_ok=True)
# trans_n = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# trans_crop = transforms.CenterCrop(128)
# trnas_resize = transforms.Resize(64)
# trans_tensor = transforms.ToTensor()
# trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])
# # trans = transforms.Compose([trnas_resize, trans_tensor])
# dataset = torchvision.datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
# torch.manual_seed(0)
# length = len(dataset)
# dataset, _ = torch.utils.data.random_split(dataset, [data_num, length-data_num])
# for imgid in range(len(dataset)):
#     torchvision.utils.save_image(dataset[imgid][0], os.path.join(result_dir, f'{imgid}.png'))
# exit()


netG = models.GanGenerator(z_dim=NZ, y_dim=2).cuda()
testIter = [1000,2000,3000,4000,5000,10000,20000,30000]
# testIter = [1000,2000,3000,4000,5000,10000]
netG.eval()
result_dir = './fakeimg/pargan'
os.makedirs(result_dir, exist_ok=True)
for it in testIter:
    print("Iter", it)
    netG.load_state_dict(torch.load("./models/zdim256/datasetSize4096/PARGAN_model/discNum4/params/G_{}.pkl".format(it)))
    randz = torch.randn((4096,NZ)).cuda()
    randc = torch.randint(YD,(4096,))
    randc = torch.nn.functional.one_hot(randc, YD).cuda()
    fake = netG(randz,randc)
    local_dir = os.path.join(result_dir,f"iter{it}")
    os.makedirs(local_dir, exist_ok=True)
    for imgid in range(fake.shape[0]):
        torchvision.utils.save_image(fake[imgid], os.path.join(local_dir,f'{imgid}.png'))

