import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

Path = os.getcwd()
print(Path)

EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")
print(EFD)
print(type(EFD))
npEFD = (EFD.to_numpy())

EFD_vals = npEFD[:,1:]
print(EFD_vals)

CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")
# print(EFD)
print(type(CPD))
npCPD = (CPD.to_numpy())

np.random.shuffle(npCPD)

CPD_comp = npCPD[:,1:-2]
print(CPD_comp)

CPD_op = npCPD[:,-2:]
print(CPD_op)

fmv = np.zeros((27,69,2))
print(fmv)

def get_al_factor(al_num, feat_num):
    al = CPD_comp[al_num]
    num = 0
    denom = 0
    for i, ele_comp in enumerate(al):
        num += ele_comp * EFD_vals[feat_num][i]
        denom += ele_comp
    key_mean = num/denom

    num = 0
    for i, ele_comp in enumerate(al):
        num += ele_comp * ((EFD_vals[feat_num][i] - key_mean)**2)
    key_vari = num/denom
    return key_mean, key_vari

for al in range(27):
    for feat in range(69):
        fmv[al][feat][0], fmv[al][feat][1] = get_al_factor(al, feat)

print(fmv)

split = 22

# train_set = npCPD[:split]
# test_set  = npCPD[split:]

# train_comp = train_set[:,1:-2]
# train_prop = train_set[:,-2:]

# test_comp = test_set[:,1:-2]
# test_prop = test_set[:,-2:]

# lin_fmv = fmv.reshape((fmv.size,1))

# print(lin_fmv)

lin_fmv = fmv.reshape((27,138))

train_fmv = lin_fmv[:split, :]
test_fmv = lin_fmv[split:,:]

avg_fmv = np.mean(train_fmv, axis = 0)

print(train_fmv.shape)
print(avg_fmv)

r = np.zeros((138,138))

for i in range(138):
    # r[i][i] = 1
    for j in range((i+1),138):
        num = 0
        denomi = 1e-9
        denomj = 1e-9
        for al in range(split):
            num += (train_fmv[al][i] - avg_fmv[i]) * (train_fmv[al][j] - avg_fmv[j])
            denomi += (train_fmv[al][i] - avg_fmv[i])**2
            denomj += (train_fmv[al][j] - avg_fmv[j])**2
        r[i][j] = num/(math.sqrt(denomi) * math.sqrt(denomj))
        r[j][i] = r[i,j]

plt.matshow(r)
plt.colorbar()
plt.show()

Selects1 = list()

for x in range(138):
    for y in range(x,138):
        if abs(r[x][y]) > 0.95:
            print(x, y)
            Selects1.append(x)

print(set(Selects1))