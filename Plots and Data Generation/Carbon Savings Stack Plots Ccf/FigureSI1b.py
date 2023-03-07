from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def SweepGeneation(NG,C_Start,C_End,S_Start,S_End):

    steps = 100
    Conv = np.linspace(C_Start,C_End, steps)
    Scav = np.linspace(S_Start,S_End, steps)

    Gen = np.ndarray(shape=(len(NG.Mix['Technologies']), steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        DNG = Dispatch(NG)
        DNG = DNG.run(Conv[idx]+1, Scav[idx])

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            Gen[jdx][idx] = Gen[jdx][idx] +  np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
        if idx == 0:
            orgco = Gen[:,0].copy()
        Gen[:,idx] = (orgco - Gen[:,idx])

    return Scav, Gen

grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM')
device_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle24u.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
Conv, Gen = SweepGeneation(NG, 0, 0, 0, 2)
DNG = Dispatch(NG)
FontSize = 14
plt.rcParams["figure.figsize"] = (4, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
Gen = Gen[0:2]
Gen = Gen[::-1]
print(np.sum(Gen,axis=0)[50])
plt.stackplot(Conv, Gen[0:2,:])
plt.xlabel('C$_{CF}$',fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.xlim(left=0,right=2)
plt.ylabel('Annual CO$_2$e Emissions Savings (Mt)',fontsize=FontSize)
plt.ylim(bottom=0,top=40)
plt.yticks(fontsize=FontSize)
#plt.legend(loc='upper left')
#plt.suptitle('b)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('S3c.png',transparent=True)
plt.savefig('S3c.svg',transparent=True)
#plt.show()