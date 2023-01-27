import numpy as np
import pandas as pd
import os

Path = os.getcwd()
print(Path)
EFD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")
# print(EFD)
print(type(EFD))
npEFD = (EFD.to_numpy())
EFD_vals = npEFD[:,1:]
print(EFD_vals)

CPD = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")
# print(EFD)
print(type(CPD))
npCPD = (CPD.to_numpy())
CPD_comp = npCPD[:,1:-2]
print(CPD_comp)
CPD_op = npCPD[:,-2:]
print(CPD_op)
