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
        NG = Scaling(NG, Conv[idx]+1, Scav[idx])
        DNG = Dispatch(NG)

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            Gen[jdx][idx] = np.sum(Asset['Generation'] / 1000000 / 2)

    return Conv, Gen


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_No_BTM.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\DSSC.csv', 53.13359, -1.746826)
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

plt.text(0.1,100,'Fossil Gas',c='white')
plt.text(0.1,10,'Fossil Hard Coal',c='white')
plt.text(0.1,235,'Nuclear', c='white')
#plt.text(1.6,280, 'SolarBTM', c='white')
plt.text(1.6,255, 'Solar', c='white')
plt.annotate('Wind Onshore', xy=(1,185),xycoords='data', xytext=(1.1,115),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Wind Offshore', xy=(0.95,170),xycoords='data', xytext=(1.1,105),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Run-of-River', xy=(0.9,163.25),xycoords='data', xytext=(1.1,95),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Pumped Storage', xy=(0.85,161),xycoords='data', xytext=(1.1,85),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')

plt.xlabel('C$_C$',fontsize=FontSize)
plt.xlim(left=0,right=2)
plt.ylabel('Energy Generated (TWh)',fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.xticks(np.arange(0,2.25,0.25)[::2],fontsize=FontSize)
plt.suptitle('a)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('Figure2aBangorNoBTM.png',Transparent=True)
plt.savefig('Figure2aBangorNoBTM.svg',Transparent=True)
plt.show()