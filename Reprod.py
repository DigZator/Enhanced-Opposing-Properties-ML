import numpy as np
import pandas as pd
import os

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

fmv = np.zeros((69,2))
print(fmv)

split = 22

for i in range(69):
    # Calculating Mean
    num = 0
    for j in range(7):
        for alloy in range(split):
            num += (EFD_vals[i][j]*CPD_comp[alloy][j])
    denom = np.sum(CPD_comp[:][:])
    fmv[i][0] = num/denom
    
    # Calculating Variance
    num = 0
    for j in range(7):
        for alloy in range(split):
            num += (((EFD_vals[i][j] - fmv[i][0])**2) *CPD_comp[alloy][j])
    denom = np.sum(CPD_comp[:][:])
    fmv[i][1] = num/denom

print(num, denom)
print(fmv)

train_set = npCPD[:22]
test_set  = npCPD[22:]

train_comp = train_set[:,1:-2]
train_prop = train_set[:,-2:]

test_comp = test_set[:,1:-2]
test_prop = test_set[:,-2:]

# lin_fmv = fmv.reshape((fmv.size,1))

# print(lin_fmv)

def get_al_factor(al_comp, feat_num):
    al = train_comp[al_comp]
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