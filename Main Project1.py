#Project1

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Path = os.getcwd() #Checking for path
#print(Path)

#Reading Data
EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")#reading data set features
#print(EFD)
#print(type(EFD))

#creating a numpy array
npEFD = (EFD.to_numpy())
#print(npEFD.shape)

#changing the shape of the array
EFD_vals = npEFD[:,1:]
#print(EFD_vals)

#Again repeating same thing for Composition
CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")#reading data set properties
#print(EFD)
#print(type(CPD))
#print(CPD)
npCPD = (CPD.to_numpy())
#print(npCPD)
np.random.shuffle(npCPD)#randomizing i dontknow why

CPD_comp = npCPD[:,1:-2]
print(CPD_comp)

CPD_op = npCPD[:,-2:]
print(CPD_op)

fmv = np.zeros((27,69,2))
print(fmv)