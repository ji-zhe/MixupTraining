import json
import numpy as np
mode = 'base'
f = open(f'./jsonfile/{mode}_scoreAll.json')
scoreAll = json.load(f)
f.close()

f = open('./jsonfile/mask.json','r')
fullMask = json.load(f)
f.close()
fullMask = fullMask[:128]
scoreAll = np.array(scoreAll)
fullMask = np.array(fullMask).T

inLoss = []
outLoss = []
for i in range(10000):
    Lin = scoreAll[fullMask[i],i].tolist()
    Lout = scoreAll[np.logical_not(fullMask[i]),i].tolist()
    inLoss.append(Lin)
    outLoss.append(Lout)
f = open(f'./jsonfile/{mode}_inLoss.json', 'w')
json.dump(inLoss,f)
f.close()
f = open(f'./jsonfile/{mode}_outLoss.json','w')
json.dump(outLoss, f)
f.close()