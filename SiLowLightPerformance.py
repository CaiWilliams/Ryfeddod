import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def IrradianceDependance(File):
    JV = pd.read_csv(File)
    JV1000 = JV.iloc[1:, 0:2].set_axis(['Voltage','Current'], axis=1, inplace=False).dropna().astype(float)
    JV800 = JV.iloc[1:, 2:4].set_axis(['Voltage','Current'], axis=1, inplace=False).dropna().astype(float)
    JV600 = JV.iloc[1:, 4:6].set_axis(['Voltage','Current'], axis=1, inplace=False).dropna().astype(float)
    JV400 = JV.iloc[1:, 6:8].set_axis(['Voltage','Current'], axis=1, inplace=False).dropna().astype(float)
    JV200 = JV.iloc[1:, 8:10].set_axis(['Voltage','Current'], axis=1, inplace=False).dropna().astype(float)

    X = np.array([200,400,600,800,1000])
    Y = np.zeros(len(X))

    Y[0] = max(JV200['Current']*JV200['Voltage'])/(200*1.73)
    Y[1] = max(JV400['Current']*JV400['Voltage'])/(400*1.73)
    Y[2] = max(JV600['Current']*JV600['Voltage'])/(600*1.73)
    Y[3] = max(JV800['Current']*JV800['Voltage'])/(800*1.73)
    Y[4] = max(JV1000['Current']*JV1000['Voltage'])/(1000*1.73)
    return X,Y

X1, Y1 = IrradianceDependance('Misc Data/LGNEORJV.csv')
X2, Y2 = IrradianceDependance('Misc Data/LGNeoN2Black.csv')
X3, Y3 = IrradianceDependance('Misc Data/LGMonoX.csv')
#plt.plot((X1+X2+X3)/3, (Y1+Y2+Y3)/3)
plt.show()