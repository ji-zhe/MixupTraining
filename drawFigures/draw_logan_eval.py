from matplotlib import pyplot as plt
import numpy as np
from utils import jload

modelID = 0
mode = 'reLaxLoss_changeAllSign'
score = jload(f"./jsonfile/logan/{modelID}/atkScore_{mode}.json")
score = np.array(score)
score_pos = score[score[:,1]==1]
score_neg = score[score[:,1]==0]

plt.hist(score_pos[:,0], bins=100, density=True, label='pos', alpha=0.5)
plt.hist(score_neg[:,0], bins=100, density=True, label='neg', alpha=0.5)
plt.legend()
plt.savefig(f"./logan_hist_{mode}Model{modelID}.png")