from matplotlib import pyplot as plt
import numpy as np
from utils import jload

modelID = 0
mode = 'base'
for mode in ['relaxLoss_changeAllSign','pargan']:
    score = jload(f"./jsonfile/atk_ratio/{modelID}/atk_ratio_{mode}.json")
    score = np.array(score)
    # import pdb;pdb.set_trace()
    score_pos = score[score[:,2]> 0.9][:,0]
    score_neg = score[score[:,2]< 0.1][:,0]

    score_pos = np.log(score_pos)
    score_neg = np.log(score_neg)

    plt.hist(score_pos, bins=100, density=True, label='pos', alpha=0.5)
    plt.hist(score_neg, bins=100, density=True, label='neg', alpha=0.5)
    plt.legend()
    plt.savefig(f"./ratio_hist_{mode}Model{modelID}.png")
    plt.close()