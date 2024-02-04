from init import *

Path = os.getcwd()
print(Path)

# Reading the Elemental property data
EFD = pd.read_excel(Path+"\Al_Elemental_Data.xlsx")

npEFD = (EFD.to_numpy())

EFD_vals = npEFD[:,1:]
# print(EFD_vals)

# Reading the composition and properties data
CPD = pd.read_excel(Path+"\Al_Alloys.xlsx")
# print(EFD)
# print(type(CPD))
npCPD = (CPD.to_numpy())

# Separating the composition and properties into differnt arrays
CPD_comp = npCPD[:,1:-2]
CPD_op = npCPD[:,-2:]
# print(CPD_comp)
# print(CPD_op)
# print(CPD_op.shape)
# print(npEFD.shape)

# exit()

n_alloys = CPD_op.shape[0]
n_prop = npEFD.shape[0]
# n_prop = 5
# Creating the Feature 
fmv = np.zeros((n_alloys, n_prop, 2))
# print(fmv)

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
for al in range(n_alloys):
    for feat in range(n_prop):
        fmv[al][feat][0], fmv[al][feat][1] = get_al_factor(al, feat)

split = n_alloys - n_alloys//5

train_set = npCPD[:split]
test_set  = npCPD[split:]

train_comp = train_set[:,1:-2]
train_prop = train_set[:,-2:]

test_comp = test_set[:,1:-2]
test_prop = test_set[:,-2:]

lin_fmv = fmv.reshape((n_alloys, n_prop*2))
train_fmv = lin_fmv[:split, :]
test_fmv = lin_fmv[split:,:]

# Calculate the average alloy factors
avg_fmv = np.mean(train_fmv, axis = 0)

# print(fmv)

t_prop = n_prop*2
# Calculate the Pearson Correlation coefficient r
r = np.zeros((t_prop, t_prop))
for i in range(t_prop):
    # r[i][i] = 1
    for j in range((i+1), t_prop):
        num = 0
        denomi = 1e-9
        denomj = 1e-9
        for al in range(split):
            num += (train_fmv[al][i] - avg_fmv[i]) * (train_fmv[al][j] - avg_fmv[j])
            denomi += (train_fmv[al][i] - avg_fmv[i])**2
            denomj += (train_fmv[al][j] - avg_fmv[j])**2
        # print(num)
        r[i][j] = num/(math.sqrt(denomi) * math.sqrt(denomj))
        r[j][i] = r[i,j]

# print(r[8,9])

plt.matshow(r)
plt.colorbar()
plt.show()

# Creating Property Labels
Prop_lab = []
for line in npEFD[:,0]:
    code = line.split()[0]
    # Prop_lab.append(code+'-M')
    # Prop_lab.append(code+'-V')
    Prop_lab.append('M-'+ code)
    Prop_lab.append('V-'+ code)

# print(len(Prop_lab), Prop_lab)

# Performing Screening Correlation
meth = 1
accountedfor = set()
sel = list()

if meth == 0:
    for x in range(t_prop):
        if x in accountedfor:
            continue
        accountedfor.add(x)
        for y in range(x, t_prop):
            if abs(r[x][y]) > 0.95:
                accountedfor.add(y)
        sel.append(x)

elif meth == 1:
    for x in range(t_prop):
        if x in accountedfor:
            continue
        accountedfor.add(x)
        select_x = True  # Added a flag to keep track of x selection
        for y in range(x+1, t_prop):  # Start from x+1 to avoid comparing to itself
            if abs(r[x][y]) > 0.95:
                accountedfor.add(y)
                select_x = False  # Set the flag to False when the condition is met
        if select_x:
            sel.append(x)

elif meth == 2:
    p = np.triu(np.corrcoef(train_fmv, rowvar=False), k=1)
    corr_threshold = 0.95
    corr_matrix = np.abs(p)

    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    highly_correlated = (corr_matrix > corr_threshold) & mask
    sel = [i for i in range(len(p)) if i not in set(np.where(highly_correlated.sum(axis=1) > 0)[0])]

elif meth == 3:
    corr = {i: [0, []] for i in range(r.shape[0])}

    # Performing Screening Correlation
    for i in range(r.shape[0]):
        for j in range(i, r.shape[1]):
            if r[i][j] > 0.95:
                corr[i][0] += 1
                corr[i][1].append(j)

    result = [key for key, value in sorted(corr.items(), key=lambda item: item[1][0], reverse=True) if value[0] > 0]

    sel = list(range(r.shape[0]))

    while len(result) > 0:
        for i in corr[result[0]][1]:
            if i in result:
                result.remove(i)
            if i in sel:
                sel.remove(i)
        result.pop(0)

print(sel)
plt.matshow(r[sel,:][:,sel])
plt.colorbar()
plt.show()

sel_lab = []

for keynum in (sel):
    sel_lab.append(Prop_lab[keynum])
print("***Correlation Screening Selects***")
print(sel_lab)

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

# Data Normalization
scaler = StandardScaler()
scaler.fit(train_fmv)
nor_train_fmv = scaler.transform(train_fmv)
nor_test_fmv = scaler.transform(test_fmv)

# Train Full model
print("\n*****\nFull Model")
svrUTS = SVR()
svrUTS.fit(nor_train_fmv, train_prop[:,0])
test_op_UTS = svrUTS.predict(nor_test_fmv)
train_op_UTS = svrUTS.predict(nor_train_fmv)

svrEC = SVR()
svrEC.fit(nor_train_fmv, train_prop[:,1])
test_op_EC = svrEC.predict(nor_test_fmv)
train_op_EC = svrEC.predict(nor_train_fmv)

print("UTS")
print("Train")
printscore(svrUTS, nor_train_fmv, train_prop[:,0])
print("\nTest")
printscore(svrUTS, nor_test_fmv, test_prop[:,0])

print("\n\nEC")
print("Train")
NCUTS, _, _, _ = printscore(svrEC, nor_train_fmv, train_prop[:,1])
print("Test")
NCEC, _, _, _ = printscore(svrEC, nor_test_fmv, test_prop[:,1])
print("*****")

print("\n*****\nAfter Correlation Screening")
noncor_train = nor_train_fmv[:,sel]
noncor_test = nor_test_fmv[:,sel]
noncormodUTS  = SVR()
noncormodUTS.fit(noncor_train, train_prop[:,0])
print("UTS\nTrain")
printscore(noncormodUTS, noncor_train, train_prop[:,0])
print("\nTest")
printscore(noncormodUTS, noncor_test, test_prop[:,0])

noncormodEC  = SVR()
noncormodEC.fit(noncor_train, train_prop[:,1])
print("\nEC\nTrain")
printscore(noncormodEC, noncor_train, train_prop[:,1])
print("\nTest")
printscore(noncormodEC, noncor_test, test_prop[:,1])
print("*****")

end = False
UTS_RE = noncor_train.copy()
EC_RE = noncor_train.copy()
UTS_RE_SEL = sel.copy()
EC_RE_SEL = sel.copy()

def smallest_value(input_dict):
    if not input_dict:
        return []
    min_first_value = min(input_dict.values(), key=lambda x: x[0])[0]
    keys = [key for key, (value1, value2) in input_dict.items() if value1 == min_first_value]
    return keys

UTSscore = {}

print("\nUTS RFE Scores")
for n_sel in range(1, len(UTS_RE_SEL)):
    modsvr = SVR(kernel = 'linear')
    modrfe = RFE(estimator = modsvr, n_features_to_select = n_sel)
    modrfe.fit(UTS_RE, train_prop[:,0])
    
    xtrain = modrfe.transform(UTS_RE)
    xtest = modrfe.transform(noncor_test)

    modsvr.fit(xtrain, train_prop[:,0])

    # print("Train Error")
    _, mtra, _, _ = printscore(modsvr, xtrain, train_prop[:, 0], printtrue = False)
    # print("Test Error")
    _, mtes, _, _ = printscore(modsvr, xtest, test_prop[:, 0], printtrue = False)

    UTSscore[n_sel] = [mtra, mtes]

for i in UTSscore.keys():
    print(f"{i} - {UTSscore[i]}")

UTS_nsel = smallest_value(UTSscore)[0]
print("Number of Features for best performance - ", UTS_nsel)

plt.plot(UTSscore.keys(), [UTSscore[i][0] for i in UTSscore.keys()], linestyle = "dashed", linewidth = 1, color = "gray", label = "line", zorder = 0)
plt.scatter(UTSscore.keys(), [UTSscore[i][0] for i in UTSscore.keys()], marker = "o", color = "blue", label = "points", zorder = 1)
plt.xlabel("Variation of Alloy Factor Number during Elimination")
plt.ylabel("Mean Absolute Error")
plt.title("Recurrsive Feature Elimination for UTS")
plt.show()

ECscore = {}

print("\nEC RFE Scores")
for n_sel in range(1, len(EC_RE_SEL)):
    modsvr = SVR(kernel = 'linear')
    modrfe = RFE(estimator = modsvr, n_features_to_select = n_sel)
    modrfe.fit(EC_RE, train_prop[:,1])
    
    xtrain = modrfe.transform(EC_RE)
    xtest = modrfe.transform(noncor_test)

    modsvr.fit(xtrain, train_prop[:,1])

    # print("Train Error")
    _, mtra, _, _ = printscore(modsvr, xtrain, train_prop[:, 1], printtrue = False)
    # print("Test Error")
    _, mtes, _, _ = printscore(modsvr, xtest, test_prop[:, 1], printtrue = False)

    ECscore[n_sel] = [mtra, mtes]

for i in ECscore.keys():
    print(f"{i} - {ECscore[i]}")

EC_nsel = smallest_value(ECscore)[0]
print("Number of Features for best performance - ", EC_nsel)

plt.plot(ECscore.keys(), [ECscore[i][0] for i in ECscore.keys()], linestyle = "dashed", linewidth = 1, color = "gray", label = "line", zorder = 0)
plt.scatter(ECscore.keys(), [ECscore[i][0] for i in ECscore.keys()], marker = "o", color = "blue", label = "points", zorder = 1)
plt.xlabel("Variation of Alloy Factor Number during Elimination")
plt.ylabel("Mean Absolute Error")
plt.title("Recurrsive Feature Elimination for EC")
plt.show()

UTSmod = SVR(kernel = "linear")
UTSmodrfe = RFE(estimator = UTSmod, n_features_to_select = UTS_nsel)
UTSmodrfe.fit(noncor_train, train_prop[:, 0])
UTSsupp = (UTSmodrfe.support_)
# print(UTSsupp)

# print(len(UTSsupp))
# print(nor_train_fmv.shape)
# print(nor_test_fmv.shape)
# print(len(UTS_RE_SEL))

UTSrefsel = []
for i in range(len(UTSsupp)):
    if UTSmodrfe.support_[i]:
        UTSrefsel.append(UTS_RE_SEL[i])

ECmod = SVR(kernel = "linear")
ECmodrfe = RFE(estimator = ECmod, n_features_to_select = EC_nsel)
ECmodrfe.fit(noncor_train, train_prop[:, 1])
ECsupp = (ECmodrfe.support_)
# print(ECsupp)

ECrefsel = []
for i in range(len(ECsupp)):
    if ECmodrfe.support_[i]:
        ECrefsel.append(EC_RE_SEL[i])
print("\nUTS RFE Selected Features")
print(UTSrefsel)
for i in UTSrefsel:
    print(Prop_lab[i])
print("\nEC RFE Selected Features")
print(ECrefsel)
for i in ECrefsel:
    print(Prop_lab[i])

UTScombs = []
for r in range(1, len(UTSrefsel)+1):
    UTScombs.extend(list(it.combinations(UTSrefsel, r)))

ECcombs = []
for r in range(1, len(ECrefsel)+1):
    ECcombs.extend(list(it.combinations(ECrefsel, r)))

UTStrain = UTSmodrfe.transform(noncor_train)
ECtrain = ECmodrfe.transform(noncor_train)

errdat = []
xdat = []

for i in range(len(UTScombs)):
    xdat.append(len(UTScombs[i]))
    take = [a in UTScombs[i] for a in UTSrefsel]
    take = np.where(take)[0]
    # print(take)
    model = SVR(kernel = "linear")
    model.fit(UTStrain[:, take], train_prop[:,0])
    _, err, _, _ = printscore(model, UTStrain[:, take], train_prop[:,0], False)
    errdat.append(err)

plt.scatter(xdat, errdat)
plt.xlabel("Variation of Alloy Factor Number during Exhaustive Screening")
plt.ylabel("Mean Absolute Error")
plt.title("Exhaustive Screening for UTS")
plt.show()

best_combUTS = UTScombs[np.argmin(errdat)]
print("\nUTS Exhaustive Screening Best Combination")
print(best_combUTS)
for i in range(len(best_combUTS)):
    print(Prop_lab[best_combUTS[i]])

errdat = []
xdat = []

for i in range(len(ECcombs)):
    xdat.append(len(ECcombs[i]))
    take = [a in ECcombs[i] for a in ECrefsel]
    take = np.where(take)[0]
    # print(take)
    model = SVR(kernel = "linear")
    model.fit(ECtrain[:, take], train_prop[:,1])
    _, err, _, _ = printscore(model, ECtrain[:, take], train_prop[:,1], False)
    errdat.append(err)

plt.scatter(xdat, errdat)
plt.xlabel("Variation of Alloy Factor Number during Exhaustive Screening")
plt.ylabel("Mean Absolute Error")
plt.title("Exhaustive Screening for EC")
plt.show()

best_combEC = ECcombs[np.argmin(errdat)]
print("\nEC Exhaustive Screening Best Combination")
print(best_combEC)
for i in range(len(best_combEC)):
    print(Prop_lab[best_combEC[i]])

print("\n\nSVR\n***UTS***")
finUTS = SVR(kernel = "linear")
finUTSTrain = UTSmodrfe.transform(noncor_train)
finUTSTest = UTSmodrfe.transform(noncor_test)
UTStake = [a in best_combUTS for a in UTSrefsel]
UTStake = np.where(UTStake)[0]
finUTS.fit(finUTSTrain[:, UTStake], train_prop[:,0])
print("Training Score")
printscore(finUTS, finUTSTrain[:, UTStake], train_prop[:,0])
print("\nTesting Score")
printscore(finUTS, finUTSTest[:, UTStake], test_prop[:,0])

plt.scatter(finUTS.predict(finUTSTest[:, UTStake]), test_prop[:,0], c = 'red', label = 'Testing Set')
plt.scatter(finUTS.predict(finUTSTrain[:, UTStake]), train_prop[:,0], c = 'blue', label = 'Training Set')
plt.legend()
plt.plot([50,400], [50,400], c = "gray")
plt.xlabel("Predicted UTS (MPa)")
plt.ylabel("Experimental UTS (MPa)")
plt.title("The performance of the final UTS model")
plt.show()

print("***EC***")
finEC = SVR(kernel = "linear")
finECTrain = ECmodrfe.transform(noncor_train)
finECTest = ECmodrfe.transform(noncor_test)
ECtake = [a in best_combEC for a in ECrefsel]
ECtake = np.where(ECtake)[0]
finEC.fit(finECTrain[:, ECtake], train_prop[:,1])
print("Training Score")
printscore(finEC, finECTrain[:, ECtake], train_prop[:,1])
print("\nTesting Score")
printscore(finEC, finECTest[:, ECtake], test_prop[:,1])

plt.scatter(finEC.predict(finECTrain[:, ECtake]), train_prop[:,1], c = 'blue', label = 'Training Set')
plt.scatter(finEC.predict(finECTest[:, ECtake]), test_prop[:,1], c = 'red', label = 'Testing Set')
plt.legend()
plt.plot([0,100], [0,100], c = "gray")
plt.xlabel("Predicted EC (%IACS)")
plt.ylabel("Experimental EC (%IACS)")
plt.title("The performance of the final EC model")
plt.show()