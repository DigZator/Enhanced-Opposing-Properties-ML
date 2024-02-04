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
# np.random.shuffle(npCPD)

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

Prop_lab = []
for line in npEFD[:,0]:
    code = line.split()[0]
    # Prop_lab.append(code+'-M')
    # Prop_lab.append(code+'-V')
    Prop_lab.append('M-'+ code)
    Prop_lab.append('V-'+ code)


# for i in range(len(Prop_lab)):
#     print(i, Prop_lab[i])

sel = [28, 39, 115, 135]
# sel = [73, 93, 105, 34, 61, 113, 47, 85, 93]
# sel = [i for i in range(len(Prop_lab))]

sel_lab = []

for keynum in (sel):
    sel_lab.append(Prop_lab[keynum])
print(sel_lab)

train_sel = train_fmv[:,sel]
test_sel = test_fmv[:,sel]

scaler = StandardScaler()
train_sel = scaler.fit_transform(train_sel)
test_sel = scaler.transform(test_sel)

nind = [i for i in range(len(sel))]
combs = list(it.combinations(nind, 2)) + list(it.combinations(nind, 3))

print(combs)

scoreUTS = {}

for i, comb in enumerate(combs):
    print(comb)
    combtrain = train_sel[:,comb]
    combtest = test_sel[:,comb]
    mod = SVR(kernel = "linear", gamma = 0.0085, C = 13)
    mod.fit(combtrain, train_prop[:, 0])
    scoreUTS[i] = printscore(mod, combtest, test_prop[:, 0], False)[0]

scoreEC = {}

for i, comb in enumerate(combs):
    combtrain = train_sel[:,comb]
    combtest = test_sel[:,comb]
    mod = SVR(kernel = "linear", gamma = 0.0011, C = 700)
    mod.fit(combtrain, train_prop[:, 1])
    scoreEC[i] = printscore(mod, combtest, test_prop[:, 1], False)[0]

barindx = np.arange(len(scoreUTS))

combine = [scoreUTS[i] + scoreEC[i] for i in range(len(scoreUTS))]

print("Combine\n", min(combine), combine.index(min(combine)), combs[combine.index(min(combine))])
print("UTS\n", min(scoreUTS.values()), combs[list(scoreUTS.values()).index(min(scoreUTS.values()))])
print("EC\n", min(scoreEC.values()), combs[list(scoreEC.values()).index(min(scoreEC.values()))])

combindx = combine.index(min(combine))

bar_width = 0.2
plt.bar(barindx - 0.1, scoreUTS.values(), bar_width, label = "UTS")
plt.bar(barindx + 0.1, scoreEC.values(), bar_width, label = "EC")
plt.xticks(barindx, scoreUTS.keys())
plt.xlabel("Combination Index")
plt.ylabel("MAPE")
plt.title("MAPE for different combinations")
plt.show()

combtrain = train_sel[:,combs[combindx]]
combtest = test_sel[:,combs[combindx]]

UTSmod = SVR(kernel = "linear", gamma = 0.0085, C = 13)
UTSmod.fit(combtrain, train_prop[:, 0])
ECmod = SVR(kernel = "linear", gamma = 0.0011, C = 700)
ECmod.fit(combtrain, train_prop[:, 1])

fig, axs = plt.subplots(1, 2)
axs[0].scatter(test_prop[:, 0], UTSmod.predict(combtest), c = "red", label = "Testing Set")
axs[0].scatter(train_prop[:, 0], UTSmod.predict(combtrain), c = "blue", label = "Training Set")
axs[0].plot([200,500], [200,500], c = "gray")
axs[0].set_xlabel("True UTS")
axs[0].set_ylabel("Predicted UTS")
axs[0].legend()
axs[0].set_title("UTS")
axs[1].scatter(test_prop[:, 1], ECmod.predict(combtest), c = "red", label = "Testing Set")
axs[1].scatter(train_prop[:, 1], ECmod.predict(combtrain), c = "blue", label = "Training Set")
axs[1].plot([0, 80], [0, 80], c = "gray")
axs[1].set_xlabel("True EC")
axs[1].set_ylabel("Predicted EC")
axs[1].legend()
axs[1].set_title("EC")

# plt.tight_layout()
plt.show()