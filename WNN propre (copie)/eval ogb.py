import pandas as pd
import numpy as np


n = 10
aucval =[]
auctest = []
for i in range(n):
    df = pd.read_csv("data/ogbtest"+str(i)+".dat")
    ind = df["aucval"].argmax()
    aucval.append(df["aucval"][ind])
    auctest.append(df["auctest"][ind])
aucval = np.array(aucval)
auctest = np.array(auctest)
print(aucval.mean(),auctest.mean())