from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def SweepGeneation(NG,C_Start,C_End,S_Start,S_End):

    steps = 100
    Conv = np.linspace(C_Start, C_End, steps)
    Scav = np.linspace(S_Start, S_End, steps)

    Gen = np.ndarray(shape=(len(NG.Mix['Technologies']), steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        DNG = Dispatch(NG)
        DNG = DNG.run(Conv[idx]+1, Scav[idx])

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            if jdx == 3:
                jdx = 2
                Gen[jdx][idx] = Gen[jdx][idx] + np.sum(Asset['Generation'] / 1000000 / 2)
                jdx = 3
            else:
                Gen[jdx][idx] = np.sum(Asset['Generation'] / 1000000 / 2)

    return Conv, Gen

grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','2016GB.NGM')
device_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle48U.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
Conv, Gen = SweepGeneation(NG,0,2,0,0)
DNG = Dispatch(NG)
FontSize = 14
plt.rcParams["figure.figsize"] = (5, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
print(labels)


E = np.zeros(len(Gen[0]))
Gen = np.vstack([Gen,E])


#idx = [1,0,9,9,9,9,2,3,4,5,7,8,6,8,8]
idx = [1,0,2,3,4,5,7,8,6,9]
plt.stackplot(Conv, Gen[idx])

plt.text(0.1,95,'Fossil Gas',c='white',fontsize=FontSize)
plt.text(0.1,5,'Fossil Hard Coal',c='white',fontsize=FontSize)
plt.text(0.1,230,'Nuclear', c='white',fontsize=FontSize)
#plt.text(1.6,280, 'SolarBTM', c='white')
plt.text(1.3,260, 'Conventional', c='white',fontsize=FontSize)
plt.annotate('Wind Onshore', xy=(1,186.5),xycoords='data', xytext=(1.1,115),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)
plt.annotate('Wind Offshore', xy=(0.95,171.5),xycoords='data', xytext=(1.1,100),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)
plt.annotate('Hydro', xy=(0.9,156.5),xycoords='data', xytext=(1.1,85),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)
#plt.annotate('Hydro Pumped Storage', xy=(0.85,161),xycoords='data', xytext=(1.1,85),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')

plt.xlabel('C$_S$',fontsize=FontSize)
plt.xlim(left=0,right=2)
plt.ylabel('Energy Dispatched (TWh)',fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.xticks(np.arange(0,2.25,0.25)[::2],fontsize=FontSize)
plt.suptitle('a)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('Figure2a.png', transparent=True)
#plt.savefig('Figure2a.svg', transparent=True)
#plt.show()