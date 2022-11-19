import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve
from matplotlib import pyplot as plt 
import numpy as np
from utils import jload

# plt.rcParams.update({'pdf.fonttype':42, 'ps.fonttype':42})

mode = 'relaxLoss_changeAllSig'
modelID = 0
attack = 'logan'
translate = {'base': 'unprotected', 'mix': 'mixup', 'relaxLoss_changeAllSign': 'relaxLoss', 'pargan': 'PAR-GAN'}
style={'base':'-', 'mix': '--', 'relaxLoss_changeAllSign':'-.','pargan':':'}
for modelID in range(1):
    print(modelID)
    for mode in ['base','mix','relaxLoss_changeAllSign','pargan']:
    # for attack in ['atk_ratio', 'logan']:
        head = 'atkScore' if attack == 'logan' else 'atk_ratio' 
        ratio_list = jload(f'./jsonfile/{attack}/{modelID}/{head}_{mode}.json')
        ratio = np.array(ratio_list)

        y_test = ratio[:,-1]
        y_score = ratio[:,0]

        AUCROC = roc_auc_score(y_test, y_score)
        fpr, tpr, thres = roc_curve(y_test, y_score)
        print("AUCROC:", AUCROC)
        plt.figure(1)
        # plt.subplots_adjust()
        plt.xlabel("False Positive Rate", fontsize=30)
        plt.ylabel("True Positive Rate", fontsize=30)
        plt.title("Log-Log", fontsize=30)
        plt.tight_layout()
        ax = plt.gca()
        ax.set_adjustable('box')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.margins(0.05, tight=True)
        plt.loglog(fpr, tpr, label=translate[mode], linestyle=style[mode], linewidth=3)
        plt.legend(fontsize=22, bbox_to_anchor=(0.55, 0), loc='lower left')

        plt.figure(2)
        # plt.subplots_adjust()
        plt.xlabel("False Positive Rate", fontsize=30)
        plt.ylabel("True Positive Rate", fontsize=30)
        plt.title("Normal", fontsize=30)
        ax = plt.gca()
        ax.set_adjustable('box')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.margins(0.05, tight=True)
        plt.plot(fpr, tpr, label=translate[mode], linestyle=style[mode], linewidth=3)
        # plt.legend(fontsize=22, bbox_to_anchor=(0.6, 0.65), loc='upper left')
    plt.figure(1)
    plt.savefig(f"./legend/ROC_curve_Model{modelID}_{attack}_loglog.pdf", bbox_inches='tight')
    plt.clf()
    plt.figure(2)
    plt.savefig(f"./nolegend/ROC_curve_Model{modelID}_{attack}.pdf", bbox_inches='tight')
    plt.clf()