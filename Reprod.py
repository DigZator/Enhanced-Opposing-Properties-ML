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

# Reading the Elemental property data
EFD = pd.read_excel(Path+"\Element_Features_Data.xlsx")
# print(EFD)
# print(type(EFD))
npEFD = (EFD.to_numpy())

EFD_vals = npEFD[:,1:]
# print(EFD_vals)

# Reading the composition and properties data
CPD = pd.read_excel(Path+"\Composition_Properties_Data.xlsx")
# print(EFD)
# print(type(CPD))
npCPD = (CPD.to_numpy())

# Shuffling the Data
# np.random.shuffle(npCPD)

# Separating the composition and properties into differnt arrays
CPD_comp = npCPD[:,1:-2]
CPD_op = npCPD[:,-2:]

# Creating the Feature 
fmv = np.zeros((27,69,2))

# Get Alloy Factors
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

# Calling alloy factors with all the alloys as the input
for al in range(27):
    for feat in range(69):
        fmv[al][feat][0], fmv[al][feat][1] = get_al_factor(al, feat)

print(fmv)

# print(fmv.shape)
# exit()

split = 22

#Split the data into train and test data
train_set = npCPD[:split]
test_set  = npCPD[split:]

train_comp = train_set[:,1:-2]
train_prop = train_set[:,-2:]

test_comp = test_set[:,1:-2]
test_prop = test_set[:,-2:]

lin_fmv = fmv.reshape((27,138))

train_fmv = lin_fmv[:split, :]
test_fmv = lin_fmv[split:,:]

# Calculate the average alloy factors
avg_fmv = np.mean(train_fmv, axis = 0)

print(train_fmv.shape)
print(avg_fmv)

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

plt.matshow(r)
plt.colorbar()
# # plt.savefig('CorrelationMatrix-20230209-1904.png', dpi = 3000)
# plt.show()

# Performing Screening Correlation
# accountedfor = set()
# sel = list()

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

mat = r[:, sel]
mat = mat[sel, :]

plt.matshow(mat)
plt.colorbar()
# plt.show()

# print(set(Selects1))
# print(npEFD[:,0])

# Creating Property Labels
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

scaler = StandardScaler()
scaler.fit(train_sel_alfeat)
train_sel_alfeat = scaler.transform(train_sel_alfeat)
test_sel_alfeat = scaler.transform(test_sel_alfeat)

non_corr_UTS = SVR(kernel = 'rbf')
non_corr_EC = SVR(kernel = 'rbf')

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

print("UTS")
_ , _, NCUTS, _ = printscore(non_corr_UTS, train_sel_alfeat, train_prop[:, 0])
print("EC")
_, _, NCEC, _ = printscore(non_corr_EC, train_sel_alfeat, train_prop[:, 1])

end = False
UTS_RE = train_sel_alfeat.copy()
UTS_RE_SEL = sel.copy()
EC_RE = train_sel_alfeat.copy()
EC_RE_SEL = sel.copy()

print(UTS_RE_SEL)

# lrmker = NCUTS

# estim = SVR(kernel = "linear")
# rfecvUTS = RFECV(estimator = estim, step = 1, cv =10)
# scaler = StandardScaler()
# train_sel_alfeat_scaled = scaler.fit_transform(train_sel_alfeat)
# rfecvUTS.fit(train_sel_alfeat_scaled, train_prop[:, 0])
# print(rfecvUTS.n_features_)
# # print(rfecvUTS.ranking_)

# estim = SVR(kernel = "linear")
# rfecvEC = RFECV(estimator = estim, step = 1, cv =10)
# rfecvEC.fit(train_sel_alfeat_scaled, train_prop[:, 1])
# print(rfecvEC.n_features_)
# # print(rfecvEC.ranking_)

# idxUTS = np.array(np.where(rfecvUTS.ranking_ == 1)[0])
# print(idxUTS)
# idxEC = np.array(np.where(rfecvEC.ranking_ == 1)[0])
# print(idxEC)

# UTS_RFE_SEL = [UTS_RE_SEL[i] for i in idxUTS]
# EC_RFE_SEL = [EC_RE_SEL[i] for i in idxEC]

# print(UTS_RFE_SEL)
# print(EC_RFE_SEL)

# print(train_sel_alfeat.shape)
# cNCUTS = NCUTS 

# # UTS RFE
# t = True
# while (t):
#     best = np.infty
#     besti = 0
#     for i in range(UTS_RE.shape[1]):
#         tdata = UTS_RE[:, np.arange(UTS_RE.shape[1]) != i]
#         # print(tdata.shape)
#         mod = SVR(kernel = "rbf")
#         mod.fit(tdata, train_prop[:, 0])
#         _, _, trerr, _ = printscore(mod, tdata, train_prop[:, 0], printtrue = False)
#         # _, _, teerr, _ = printscore(mod, test_sel_alfeat[:, [x for x in UTS_RE_SEL if x != i]], test_prop[:, 0], printtrue = False)
#         print(f"{i}\t{trerr}\t{0}\t{besti}")
#         besti = i if best > trerr else besti
#         best = trerr if best > trerr else best

#     if best < cNCUTS:
#         train_sel_alfeat = train_sel_alfeat[:, np.arange(train_sel_alfeat.shape[1]) != besti]
#         UTS_RE_SEL.pop(UTS_RE_SEL.index(UTS_RE_SEL[besti]))
#         print(UTS_RE_SEL)
#         cNCUTS = best
#     else:
#         t = False
# print(UTS_RE_SEL)

# UTS_sel_lab = []

# for keynum in set(UTS_RE_SEL):
#     UTS_sel_lab.append(Prop_lab[keynum])

# print(UTS_sel_lab)

# for afeat in UTS_RE_SEL:
#     print(afeat, Prop_lab[afeat], end=" - \t")
#     for bfeat in range(len(r)):
#         if abs(r[afeat][bfeat]) > 0.95:
#             print(bfeat, Prop_lab[bfeat], end=" ")
#     print()

score = {}
# for n_sel in range(1, len(UTS_RE_SEL)):
for n_sel in range(6, 7):
    modsvr = SVR(kernel = "linear")
    modrfe = RFE(estimator = modsvr, n_features_to_select = n_sel)
    modrfe.fit(train_sel_alfeat, train_prop[:, 0])
    print(n_sel)

    print("Selected Features: ", modrfe.support_)

    xtrain = modrfe.transform(train_sel_alfeat)
    xtest = modrfe.transform(test_sel_alfeat)

    modsvr.fit(xtrain, train_prop[:, 0])

    # ytrain = modsvr.predict(xtrain)
    # ytest = modsvr.predict(xtest)

    print("Train Error")
    _, mtra, _, _ = printscore(modsvr, xtrain, train_prop[:, 0])
    print("Test Error")
    _, mtes, _, _ = printscore(modsvr, xtest, test_prop[:, 0])

    score[n_sel] = [mtra, mtes]
    print()

print(score)

def find_keys_with_smallest_values(my_dict):
    if not my_dict:
        return []

    max_first_value = min(my_dict.values(), key=lambda x: x[0])[0]
    max_second_value = min(my_dict.values(), key=lambda x: x[1])[1]

    largest_keys = [key for key, (value1, value2) in my_dict.items() if value1 == max_first_value or value2 == max_second_value]

    return largest_keys

print(find_keys_with_smallest_values(score))

# for i in range(len(UTS_RE_SEL)):
#     if 