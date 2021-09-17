from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SweepGeneation(NG,C_Start,C_End,S_Start,S_End):

    steps = 100
    Conv = np.linspace(C_Start,C_End, steps)
    Scav = np.linspace(S_Start,S_End, steps)

    Gen = np.ndarray(shape=(len(NG.Mix['Technologies']), steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        NG = Scaling(NG, Conv[idx]+1, Conv[idx]+1, Scav[idx], Scav[idx])
        DNG = Dispatch(NG)

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            Gen[jdx][idx] = Gen[jdx][idx] +  np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
        if idx == 0:
            orgco = Gen[:,0].copy()
        Gen[:,idx] = (orgco - Gen[:,idx])

    return Conv, Gen

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
Conv, Gen = SweepGeneation(NG,0,2,0,0)
DNG = Dispatch(NG)
plt.rcParams["figure.figsize"] = (4, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
plt.stackplot(Conv, Gen[0:2,:])
plt.xlabel('C$_C$')
plt.xlim(left=0,right=2)
plt.ylabel('CO$_2$e Emissions Savings (Mt)')
#plt.legend(labels)
plt.tight_layout()
plt.savefig('FigureSI1a.png')
plt.savefig('FigureSI1a.svg')
plt.show()