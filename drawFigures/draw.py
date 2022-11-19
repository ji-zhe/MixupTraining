from matplotlib import pyplot as plt 
import numpy as np
import torch

c1 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/base.pkl')
c2 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/pargan_discNum4_comem5_max.pkl')
c3 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/mix.pkl')
c4 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/1g2d_D1.pkl')
c5 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/1g2d_D2.pkl')
# c1 = torch.load('/home/jizhe/jz/wgan_torch_celebA/logan_result/zdim256/datasetSize4096/base.pkl')
# c2 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/base.pkl')
# c3 = torch.load('/home/jizhe/jz/wgan_torch_celebA/comem_logan_result/zdim256/datasetSize4096/comem10/base.pkl')

x = list(range(500, 30001,500))

plt.ylim(0.4,1)
plt.plot(x, c1, 'o-', label='original GAN')
plt.plot(x, c2, 'v-', label='PAR-GAN')
plt.plot(x, c3, 's-', label='Mixup')
plt.plot(x, c4, 'p-', label='Mixup Plus D1')
plt.plot(x, c5, '*-', label='Mixup Plus D2')
# plt.plot(x,c1,'o-',label='original Logan')
# plt.plot(x,c2,'v-',label='co-member strength = 5')
# plt.plot(x,c3,'s-',label='co-member strength = 10')
plt.xlabel('Num of Iteration')
plt.ylabel('Attack AUCROC')
# plt.plot(epochlist, aucprclist, label='aucprc')
plt.legend()
plt.savefig(f"comem_logan_vs_all.png")