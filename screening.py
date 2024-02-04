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

plt.matshow(r)
plt.colorbar()
# # plt.savefig('CorrelationMatrix-20230209-1904.png', dpi = 3000)
plt.show()

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
print(len(sel))

for i in sel:
    print(corr[i])