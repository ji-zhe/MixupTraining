import numpy as np
import json

a = np.load('gan_leaks_results/wb/pargan_dnum1000/pos_loss.npy')
b = np.load('gan_leaks_results/wb/pargan_dnum1000/neg_loss.npy')
o = np.ones(a.shape)
z = np.zeros(b.shape)

posPart = np.vstack((-a, o)).T
negPart = np.vstack((-b, z)).T

whole = np.vstack((posPart, negPart))
f = open('jsonfile/ganleaks/0/atk_ratio_pargan.json', 'w')
json.dump(whole.tolist(), f)
f.close()
