import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Path = os.getcwd() #Checking for path
print(Path)

EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")#reading data set features
print(EFD)
print(type(EFD))
npEFD = (EFD.to_numpy())#creating a numpy array

EFD_vals = npEFD[:,1:]#changing the shape of the array
print(EFD_vals)

CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")#reading data set properties
# print(EFD)
print(type(CPD))
npCPD = (CPD.to_numpy())

np.random.shuffle(npCPD)#randomizing

CPD_comp = npCPD[:,1:-2]
print(CPD_comp)#%ofelements

CPD_op = npCPD[:,-2:]
print(CPD_op)#property

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

train_set = npCPD[:split]
test_set  = npCPD[split:]

train_comp = train_set[:,1:-2]
train_prop = train_set[:,-2:]

test_comp = test_set[:,1:-2]
test_prop = test_set[:,-2:]

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
# plt.savefig('CorrelationMatrix-20230209-1904.png', dpi = 3000)
plt.show()
Selects1 = list()

for x in range(138):
    for y in range(x,138):
        if abs(r[x][y]) > 0.95:
            print(x, y)
            Selects1.append(x)

print(set(Selects1))
print(npEFD[:,0])
Prop_lab = []
for line in npEFD[:,0]:
    code = line.split()[0]
    # Prop_lab.append(code+'-M')
    # Prop_lab.append(code+'-V')
    Prop_lab.append('M-'+ code)
    Prop_lab.append('V-'+ code)
print(len(Prop_lab))

Selects1_lab = []

for keynum in set(Selects1):
    Selects1_lab.append(Prop_lab[keynum])

print(Selects1_lab)

# Train default model
svrUTS = make_pipeline(StandardScaler(), SVR())
svrUTS.fit(train_fmv, train_prop[:,0])
print(svrUTS.predict(test_fmv))
print(test_prop[:,0])