from init import *

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
np.random.shuffle(npCPD)

# Separating the composition and properties into differnt arrays
CPD_comp = npCPD[:,1:-2]
CPD_op = npCPD[:,-2:]
# print(CPD_comp)
# print(CPD_op)

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

def smallest_value(input_dict):
    if not input_dict:
        return []
    min_first_value = min(input_dict.values(), key=lambda x: x[0])[0]
    keys = [key for key, (value1, value2) in input_dict.items() if value1 == min_first_value]
    return keys

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

plt.matshow(r[:30,:30])
plt.colorbar()
# # plt.savefig('CorrelationMatrix-20230209-1904.png', dpi = 3000)
plt.show()

meth = 0
sel = list()

scores = {}

for meth in range(4):
    print(f"\n\n***** Method {meth} *****")
    if meth == 0:
        accountedfor = set()
        sel = list()

        for x in range(138):
            if x in accountedfor:
                continue
            accountedfor.add(x)
            for y in range(x,138):
                if abs(r[x][y]) > 0.95:
                    accountedfor.add(y)
            sel.append(x)
    elif meth == 1:
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

        print(result)

        sel = list(range(r.shape[0]))

        while len(result) > 0:
            for i in corr[result[0]][1]:
                if i in result:
                    result.remove(i)
                if i in sel:
                    sel.remove(i)
            result.pop(0)
    
    print(sel)
    sel_alfeat = lin_fmv[:,sel]
    # print(sel_alfeat.shape)


    train_sel_alfeat = sel_alfeat[:split, :]
    test_sel_alfeat = sel_alfeat[split:,:]

    scaler = StandardScaler()
    scaler.fit(train_sel_alfeat)
    train_sel_alfeat = scaler.transform(train_sel_alfeat)
    test_sel_alfeat = scaler.transform(test_sel_alfeat)
    
    ncUTS = SVR(kernel='linear', C = 13, gamma = 0.0085)
    ncEC = SVR(kernel='linear', C = 700, gamma = 0.0011)
    ncUTS.fit(train_sel_alfeat, train_prop[:,0])
    ncEC.fit(train_sel_alfeat, train_prop[:,1])

    # print("UTS")
    # printscore(ncUTS, test_sel_alfeat, test_prop[:,0], printtrue = False)
    # print("EC")
    # printscore(ncEC, test_sel_alfeat, test_prop[:,1], printtrue = False)

    UTSscore = {}
    for n_sel in range(1, len(sel)):
        modsvr = SVR(kernel='linear', C = 13, gamma = 0.0085)
        modrfe = RFE(modsvr, n_features_to_select = n_sel)
        modrfe.fit(train_sel_alfeat, train_prop[:,0])
        xtrain = modrfe.transform(train_sel_alfeat)
        xtest = modrfe.transform(test_sel_alfeat)
        modsvr.fit(xtrain, train_prop[:,0])
        # _, mtra, _, _ = printscore(modsvr, xtrain, train_prop[:, 0], printtrue = False)
        # _, mtes, _, _ = printscore(modsvr, xtest, test_prop[:, 0], printtrue = False)
        mtra, _,  _, _ = printscore(modsvr, xtrain, train_prop[:, 0], printtrue = False)
        mtes, _,  _, _ = printscore(modsvr, xtest, test_prop[:, 0], printtrue = False)
        UTSscore[n_sel] = [mtra, mtes]
    
    ECscore = {}
    for n_sel in range(1, len(sel)):
        modsvr = SVR(kernel='linear', C = 700, gamma = 0.0011)
        modrfe = RFE(modsvr, n_features_to_select = n_sel)
        modrfe.fit(train_sel_alfeat, train_prop[:,1])
        xtrain = modrfe.transform(train_sel_alfeat)
        xtest = modrfe.transform(test_sel_alfeat)
        modsvr.fit(xtrain, train_prop[:,1])
        # _, mtra, _, _ = printscore(modsvr, xtrain, train_prop[:, 1], printtrue = False)
        # _, mtes, _, _ = printscore(modsvr, xtest, test_prop[:, 1], printtrue = False)
        mtra, _, _, _ = printscore(modsvr, xtrain, train_prop[:, 0], printtrue = False)
        mtes, _, _, _ = printscore(modsvr, xtest, test_prop[:, 0], printtrue = False)
        ECscore[n_sel] = [mtra, mtes]

    UTS_nsel = smallest_value(UTSscore)[0]
    EC_nsel = smallest_value(ECscore)[0]
    print(f"UTS_nsel - {UTS_nsel}\nUTSscore - {UTSscore[UTS_nsel]}")
    print(f"EC_nsel - {EC_nsel}\nECscore - {ECscore[EC_nsel]}")

    scores[meth] = [UTS_nsel, UTSscore[UTS_nsel], EC_nsel, ECscore[EC_nsel]]

xlabels = ["Method1", "Method2", "Method3", "Method4"]
nuts = [scores[i][0] for i in range(4)]
trauts = [scores[i][1][0] for i in range(4)]
tesuts = [scores[i][1][1] for i in range(4)]
nec = [scores[i][2] for i in range(4)]
traec = [scores[i][3][0] for i in range(4)]
tesec = [scores[i][3][1] for i in range(4)]

bar_width = 0.1

index = np.arange(len(xlabels))

plt.bar(index - 2*bar_width, nuts, bar_width, label = "UTS_nsel")
plt.bar(index - bar_width, trauts, bar_width, label = "UTS_tra")
plt.bar(index, tesuts, bar_width, label = "UTS_tes")
plt.bar(index + bar_width, nec, bar_width, label = "EC_nsel")
plt.bar(index + 2*bar_width, traec, bar_width, label = "EC_tra")
plt.bar(index + 3*bar_width, tesec, bar_width, label = "EC_tes")

plt.xlabel("Method")
plt.ylabel("Error/Nsel")
plt.title("Error/Nsel for different methods")
plt.xticks(index, xlabels)
plt.legend()

def add_labels(data, offset):
    for i, value in enumerate(data):
        plt.text(index[i] + offset, value + 0.005, f'{value:.2f}', ha='center', va='bottom')

add_labels(nuts, -2*bar_width)
add_labels(trauts, -bar_width)
add_labels(tesuts, 0)
add_labels(nec, bar_width)
add_labels(traec, 2*bar_width)
add_labels(tesec, 3*bar_width)

plt.show()