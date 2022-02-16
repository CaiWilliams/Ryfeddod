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
        DNG = Dispatch(NG, Conv[idx]+1, Scav[idx])

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            Gen[jdx][idx] = Gen[jdx][idx] +  np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
        if idx == 0:
            orgco = Gen[:,0].copy()
        Gen[:,idx] = (orgco - Gen[:,idx])

    return Scav, Gen


NG = setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\NationalGrid_2016.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
Scav, Gen = SweepGeneation(NG, 0, 0, 0, 2)
DNG = Dispatch(NG, 1, 0)
plt.rcParams["figure.figsize"] = (4, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
plt.stackplot(Scav, Gen[0:2,:])
plt.xlabel('C$_S$')
plt.xlim(left=0,right=2)
plt.ylabel('CO$_2$e Emissions Savings (Mt)')
#plt.legend(labels)
plt.tight_layout()
plt.savefig('FigureSI1b.svg')
plt.savefig('FigureSI1b.png')
#plt.show()