import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Collect Data and make variables
Path = os.getcwd()
EFD = pd.read_excel(Path+"\Element_Features_Data.xlsx")
npEFD = (EFD.to_numpy())
EFD_vals = npEFD[:,1:]
CPD = pd.read_excel(Path+"\Composition_Properties_Data.xlsx")
npCPD = (CPD.to_numpy())
# np.random.shuffle(npCPD)
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

# print(train_fmv.shape)
# print(avg_fmv)

# Calculate the Pearson Correlation coefficient r
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

# Performing Screening Correlation
accountedfor = set()
sel = list()

# for x in range(138):
#     if x in accountedfor:
#         continue
#     accountedfor.add(x)
#     for y in range(x,138):
#         if abs(r[x][y]) > 0.95:
#             accountedfor.add(y)
#     sel.append(x)

accountedfor = set()
sel = list()

for x in range(138):
    if x in accountedfor:
        continue
    accountedfor.add(x)
    select_x = True  # Added a flag to keep track of x selection
    for y in range(x+1, 138):  # Start from x+1 to avoid comparing to itself
        if abs(r[x][y]) > 0.95:
            accountedfor.add(y)
            select_x = False  # Set the flag to False when the condition is met
    if select_x:
        sel.append(x)

print(sel)
print(len(sel))

sel_alfeat = lin_fmv[:, sel]

train_sel_alfeat = sel_alfeat[:split, :]
test_sel_alfeat = sel_alfeat[split:,:]

scaler = StandardScaler()
scaler.fit(train_sel_alfeat)
train_sel_alfeat = scaler.transform(train_sel_alfeat)
test_sel_alfeat = scaler.transform(test_sel_alfeat)

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}

# UTS
# param_grid = {'C': [8, 9, 10, 11, 12],
#               'gamma': [0.008, 0.009, 0.01, 0.011, 0.012]}

# param_grid = {'C': [10, 11, 12, 13, 14, 15],
#               'gamma': [0.008, 0.009, 0.01, 0.011, 0.012, 0.0095, 0.0105, 0.0115]}

param_grid = {'C': [11, 12, 13, 14, 15],
              'gamma': [0.0085, 0.009, 0.0095, 0.01, 0.0105]}


# EC
# param_grid = {'C': [800, 900, 1000, 1100, 1200],
#               'gamma': [0.0009, 0.001, 0.0011, 0.0012, 0.0013]}

# param_grid = {'C': [700, 750, 800, 850, 900],
#               'gamma': [0.0008, 0.0009, 0.001, 0.0011, 0.0012]}

scaler = StandardScaler()
scaler.fit(train_fmv)
train_fmv = scaler.transform(train_fmv)

svr = SVR()
gs = GridSearchCV(estimator = svr, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 10, n_jobs = -1)

gs.fit(train_fmv, train_prop[:, 0])
best_params = gs.best_params_
print(best_params)


## UTS - {'C': 13, 'gamma': 0.0085}
## EC - {'C': 700, 'gamma': 0.0011}