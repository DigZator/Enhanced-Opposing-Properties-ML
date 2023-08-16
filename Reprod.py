import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import RFECV, RFE

Path = os.getcwd()
print(Path)

EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")
print(EFD)
print(type(EFD))
npEFD = (EFD.to_numpy())

EFD_vals = npEFD[:,1:]
# print(EFD_vals)

CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")
# print(EFD)
print(type(CPD))
npCPD = (CPD.to_numpy())

np.random.shuffle(npCPD)

CPD_comp = npCPD[:,1:-2]
# print(CPD_comp)

CPD_op = npCPD[:,-2:]
# print(CPD_op)

fmv = np.zeros((27,69,2))
# print(fmv)

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
# # plt.savefig('CorrelationMatrix-20230209-1904.png', dpi = 3000)
# plt.show()

accountedfor = list()
sel = list()

for x in range(138):
    if x in accountedfor:
        continue
    accountedfor.append(x)
    for y in range(x,138):
        if abs(r[x][y]) > 0.95:
            accountedfor.append(y)
    sel.append(x)

print(sel)
print(len(sel))

mat = r[:, sel]
mat = mat[sel, :]

plt.matshow(mat)
plt.colorbar()
# plt.show()

# print(set(Selects1))
# print(npEFD[:,0])
Prop_lab = []
for line in npEFD[:,0]:
    code = line.split()[0]
    # Prop_lab.append(code+'-M')
    # Prop_lab.append(code+'-V')
    Prop_lab.append('M-'+ code)
    Prop_lab.append('V-'+ code)
print(len(Prop_lab))

# fig = plt.figure()
# ax = fig.add_axes([0,0,138,138])
# ax.plot(r)
# ax.set_xlabel(Prop_lab)

# ax.imshow(r)

# fig, ax = plt.subplots()
# im = ax.imshow(r)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(Prop_lab)), labels=Prop_lab, size = 8)
# ax.set_yticks(np.arange(len(Prop_lab)), labels=Prop_lab, size = 8)
# # ax.grid(color = 'w', linewidth = 2, linestyle = ':')
# plt.setp(ax.get_xticklabels(), rotation = 90, ha = 'right', rotation_mode = 'anchor')
# fig.tight_layout()
# # plt.savefig('CorrelationMatrix-20230210-0108.png', dpi = 3000)
# plt.show()

sel_lab = []

for keynum in set(sel):
    sel_lab.append(Prop_lab[keynum])

print(sel_lab)
print(len(sel_lab))

# Train Full model
svrUTS = make_pipeline(StandardScaler(), SVR())
svrUTS.fit(train_fmv, train_prop[:,0])
test_op_UTS = svrUTS.predict(test_fmv)
train_op_UTS = svrUTS.predict(train_fmv)
print(test_op_UTS)
print(test_prop[:,0])

svrEC = make_pipeline(StandardScaler(), SVR())
svrEC.fit(train_fmv, train_prop[:,1])
test_op_EC = svrEC.predict(test_fmv)
train_op_EC = svrEC.predict(train_fmv)
print(test_op_EC)
print(test_prop[:,1])

# Full Model Error
FME_UTS_train = cross_val_score(estimator = svrUTS, X = train_fmv, y = train_prop[:, 0], cv = 10, scoring = "neg_mean_absolute_percentage_error")
FME_EC_train = cross_val_score(estimator = svrEC, X = train_fmv, y = train_prop[:, 1], cv = 10, scoring = "neg_mean_absolute_percentage_error")

UTS_train_scoravg = np.average(FME_UTS_train)
EC_train_scoravg = np.average(FME_EC_train)

FME_UTS_test = cross_val_score(estimator = svrUTS, X = test_fmv, y = test_prop[:, 0], cv = 2, scoring = "neg_mean_absolute_percentage_error")
FME_EC_test = cross_val_score(estimator = svrEC, X = test_fmv, y = test_prop[:, 1], cv = 2, scoring = "neg_mean_absolute_percentage_error")

UTS_test_scoravg = np.average(FME_UTS_test)
EC_test_scoravg = np.average(FME_EC_test)

train_mae_UTS = metrics.mean_absolute_error(train_prop[:,0], train_op_UTS)
train_mse_UTS = metrics.mean_squared_error(train_prop[:,0], train_op_UTS)
train_r2_UTS = metrics.r2_score(train_prop[:,0], train_op_UTS)

train_mae_EC = metrics.mean_absolute_error(train_prop[:,1], train_op_EC)
train_mse_EC = metrics.mean_squared_error(train_prop[:,1], train_op_EC)
train_r2_EC = metrics.r2_score(train_prop[:,1], train_op_EC)

test_mae_UTS = metrics.mean_absolute_error(test_prop[:,0], test_op_UTS)
test_mse_UTS = metrics.mean_squared_error(test_prop[:,0], test_op_UTS)
test_r2_UTS = metrics.r2_score(test_prop[:,0], test_op_UTS)

test_mae_EC = metrics.mean_absolute_error(test_prop[:,1], test_op_EC)
test_mse_EC = metrics.mean_squared_error(test_prop[:,1], test_op_EC)
test_r2_EC = metrics.r2_score(test_prop[:,1], test_op_EC)

print("**********Full Model Error**********")
print(f"---svrUTS---\nTrain\nCV - {UTS_train_scoravg}\nMAE - {train_mae_UTS}\nMSE - {train_mse_UTS}\nR2 - {train_r2_UTS}\nTest\nCV  - {UTS_test_scoravg}\nMAE - {test_mae_UTS}\nMSE - {test_mse_UTS}\nR2 - {test_r2_UTS}")
print()
print(f"---svrEC---\nTrain\nCV - {EC_train_scoravg}\nMAE - {train_mae_EC}\nMSE - {train_mse_EC}\nR2 - {train_r2_EC}\nTest\nCV  - {EC_test_scoravg}\nMAE - {test_mae_EC}\nMSE - {test_mse_EC}\nR2 - {test_r2_EC}")
print("************************************")

# print(type(sel))
sel_alfeat = lin_fmv[:,sel]
print(sel_alfeat.shape)

train_sel_alfeat = sel_alfeat[:split, :]
test_sel_alfeat = sel_alfeat[split:,:]

non_corr_UTS = make_pipeline(StandardScaler(), SVR())
non_corr_EC = make_pipeline(StandardScaler(), SVR())

def printscore(model, x, y, printtrue = True):
    MAPE = metrics.mean_absolute_percentage_error(y, model.predict(x))
    MAE = metrics.mean_absolute_error(y, model.predict(x))
    MSE = metrics.mean_squared_error(y, model.predict(x))
    R2 = metrics.r2_score(y, model.predict(x))
    if printtrue:
        print(f"MAPE - {metrics.mean_absolute_percentage_error(y, model.predict(x))}")
        print(f"MAE - {metrics.mean_absolute_error(y, model.predict(x))}")
        print(f"MSE - {metrics.mean_squared_error(y, model.predict(x))}")
        print(f"R2 - {metrics.r2_score(y, model.predict(x))}")

    return MAPE, MAE, MSE, R2

non_corr_UTS.fit(train_sel_alfeat, train_prop[:,0])
non_corr_EC.fit(train_sel_alfeat, train_prop[:,1])

# printscore(svrEC, train_fmv, train_prop[:,1])
# print()
# printscore(svrEC, test_fmv, test_prop[:,1])
# print()
NCEC, _, _, _ = printscore(non_corr_EC, train_sel_alfeat, train_prop[:, 1])
# print()
# printscore(non_corr_EC, test_sel_alfeat, test_prop[:, 1])

NCUTS, _, _, _ = printscore(non_corr_UTS, train_sel_alfeat, train_prop[:, 0])
end = False
UTS_RE = train_sel_alfeat.copy()
UTS_RE_SEL = sel.copy()
EC_RE = train_sel_alfeat.copy()
EC_RE_SEL = sel.copy()

print(UTS_RE_SEL)

lrmker = NCUTS

estim = SVR(kernel = "linear")
rfecvUTS = RFECV(estimator = estim, step = 1, cv =10)
scaler = StandardScaler()
train_sel_alfeat_scaled = scaler.fit_transform(train_sel_alfeat)
rfecvUTS.fit(train_sel_alfeat_scaled, train_prop[:, 0])
print(rfecvUTS.n_features_)
print(rfecvUTS.ranking_)

estim = SVR(kernel = "linear")
rfecvEC = RFECV(estimator = estim, step = 1, cv =10)
rfecvEC.fit(train_sel_alfeat_scaled, train_prop[:, 1])
print(rfecvEC.n_features_)
print(rfecvEC.ranking_)