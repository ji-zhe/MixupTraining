from matplotlib import pyplot as plt 
import numpy as np
import torch

base = torch.load('./D_score/d_score_base13k.pkl')

MAX = max(base)
MIN = min(base)
step = (MAX - MIN) / 50
bins=[MIN + i*step for i in range(51)]
base_pos = base[:4096]
base_neg = base[4096:]

a = plt.hist(base_pos, bins=bins, alpha=0.6, label='Train')
b = plt.hist(base_neg, bins=bins, alpha=0.6, label='Holdout')
plt.legend()
plt.savefig('1g2d_d1_5k.png')
plt.clf()

# print("TVD:", np.abs(a[0]-b[0]).sum()/4096)
print("TVD:", np.abs(a[0]-b[0]).sum()/4096/2)

