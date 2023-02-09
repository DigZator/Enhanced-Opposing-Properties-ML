import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Collect Data and make variables
Path = os.getcwd()
EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")
npEFD = (EFD.to_numpy())
EFD_vals = npEFD[:,1:]
CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")
npCPD = (CPD.to_numpy())
np.random.shuffle(npCPD)
CPD_comp = npCPD[:,1:-2]
CPD_op = npCPD[:,-2:]
fmv = np.zeros((27,69,2))

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

# Split data
split = 22
train_set = npCPD[:split]
test_set  = npCPD[split:]
train_comp = train_set[:,1:-2]
train_prop = train_set[:,-2:]
test_comp = test_set[:,1:-2]
test_prop = test_set[:,-2:]

lin_fmv = fmv.reshape((27,138))
train_fmv = lin_fmv[:split, :]
test_fmv = lin_fmv[split:,:]
avg_fmv = np.mean(train_fmv, axis = 0)

print(train_fmv.shape)
print(avg_fmv)

# Train Mod
def train_model(X, Y, gamma, C):
    svr = make_pipeline(StandardScaler(), SVR(gamma = gamma, C = C))
    svr.fit(X, Y)
    # print(svr.predict(test_fmv))
    # print(test_prop[:,0])
    score = cross_val_score(estimator = svr, X = X, y = Y, cv = 10)
    return svr, score

