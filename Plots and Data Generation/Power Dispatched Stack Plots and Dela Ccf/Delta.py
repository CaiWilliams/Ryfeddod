from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def SweepGeneation(NG,C_Start,C_End,S_Start,S_End):

    steps = 100
    Conv = np.linspace(C_Start, C_End, steps)
    Scav = np.linspace(S_Start, S_End, steps)
    print(Conv)
    print(Scav)

    Gen = np.ndarray(shape=(len(NG.Mix['Technologies']), steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        DNG = Dispatch(NG,)

        DNG = DNG.run(Conv[idx]+1, Scav[idx])

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            if jdx == 3:
                jdx = 2
                Gen[jdx][idx] = Gen[jdx][idx] + np.sum(Asset['Generation'] / 1000000 / 2)
                jdx = 3
            else:
                Gen[jdx][idx] = np.sum(Asset['Generation'] / 1000000 / 2)

    return Scav, Gen


grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM')
device_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
Scav, Gen = SweepGeneation(NG,0,0,0,2)
DNG = Dispatch(NG)
FontSize = 14
plt.rcParams["figure.figsize"] = (5, 6)
plt.rcParams["figure.dpi"] = 600
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]


E = np.zeros(len(Gen[0]))
Gen = np.vstack([Gen,E])

#idx = [1,0,9,9,9,9,2,3,4,5,7,8,6,8,8]
idx = [1,0,2,6,8]

for i in idx:
    if i == 2:
        plt.plot(Scav, Gen[i] - Gen[i][0])
    elif i == 6:
        plt.plot(Scav, Gen[i] - Gen[i][0])
    elif i == 8:
        plt.plot(Scav, Gen[i] - Gen[i][0])
    else:
        plt.plot(Scav, Gen[i]-Gen[i][0])

#plt.text(0.1,5,'Fossil Hard Coal', c='white',fontsize=FontSize)
#plt.text(0.1,95,'Fossil Gas',c='white',fontsize=FontSize)
#plt.text(0.1,220,'Nuclear', c='white',fontsize=FontSize)
#plt.text(1.3,260, 'CFPV', c='white',fontsize=FontSize)
#plt.text(1.5,250, 'SolarNT', c='white')
#plt.annotate('SolarBTM', xy=(1.3,225),xycoords='data', xytext=(1.5,180),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
#plt.annotate('Conventional', xy=(1.2,258),xycoords='data', xytext=(1.3,220),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)

#plt.annotate('Wind Onshore', xy=(1,180),xycoords='data', xytext=(1.1,75),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)
#plt.annotate('Wind Offshore', xy=(0.95,165),xycoords='data', xytext=(1.1,65),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)
#plt.annotate('Hydro Run-of-River', xy=(0.9,143.5),xycoords='data', xytext=(1.1,70),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
#plt.annotate('Hydro', xy=(0.85,155),xycoords='data', xytext=(1.1,55),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white',fontsize=FontSize)

#plt.annotate('Fossil Hard Coal', xy=(0.8,120),xycoords='data', xytext=(1.1,30),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')

plt.xlabel('C$_{CF}$',fontsize=FontSize)
plt.xticks(np.arange(0,2.25,0.25)[::2],fontsize=FontSize)
plt.xlim(left=0,right=2)
#plt.ylim(top=55,bottom=-55)
plt.ylabel('Change in Energy Dispatched (TWh)',fontsize=FontSize)
plt.yticks(fontsize=FontSize)
#plt.suptitle('b)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('2i.svg',transparent=True)
plt.savefig('2i.png',transparent=True)
#plt.show()