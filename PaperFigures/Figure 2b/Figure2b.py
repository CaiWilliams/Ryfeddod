from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SweepGeneation(NG,C_Start,C_End,S_Start,S_End):

    steps = 10
    Conv = np.linspace(C_Start,C_End, steps)
    Scav = np.linspace(S_Start,S_End, steps)

    Gen = np.ndarray(shape=(len(NG.Mix['Technologies']), steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        NG = Scaling(NG, Conv[idx]+1, Conv[idx]+1, Scav[idx], Scav[idx])
        DNG = Dispatch(NG)

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            Gen[jdx][idx] = np.sum(Asset['Generation'] / 1000000 / 2)

    return Scav, Gen


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
Scav, Gen = SweepGeneation(NG,0,0,0,2)
DNG = Dispatch(NG)
FontSize = 14
plt.rcParams["figure.figsize"] = (5, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]


E = np.zeros(len(Gen[0]))
Gen = np.vstack([Gen,E])

idx = [1,0,11,11,11,11,2,3,4,5,7,11,6,8,9,10]
plt.stackplot(Scav, Gen[idx])

plt.text(0.1,7.5,'Fossil Hard Coal', c='white')
plt.text(0.1,75,'Fossil Gas',c='white')
plt.text(0.1,225,'Nuclear', c='white')
plt.text(1.5,270, 'SolarBTMNT', c='white')
plt.text(1.5,227.25, 'SolarNT', c='white')
plt.annotate('SolarBTM', xy=(1.3,225),xycoords='data', xytext=(1.5,180),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Solar', xy=(1.25,217.25),xycoords='data', xytext=(1.5,170),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Wind Onshore', xy=(1,145),xycoords='data', xytext=(1.1,70),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Wind Offshore', xy=(0.95,135),xycoords='data', xytext=(1.1,60),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Run-of-River', xy=(0.9,128.25),xycoords='data', xytext=(1.1,50),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Pumped Storage', xy=(0.85,127.5),xycoords='data', xytext=(1.1,40),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
#plt.annotate('Fossil Hard Coal', xy=(0.8,120),xycoords='data', xytext=(1.1,30),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')

plt.xlabel('C$_S$',fontsize=FontSize)
plt.xticks(np.arange(0,2.25,0.25)[::2],fontsize=FontSize)
plt.xlim(left=0,right=2)
plt.ylabel('Energy Generated (TWh)',fontsize=FontSize)
plt.yticks(fontsize=FontSize)
#plt.legend(labels)
plt.tight_layout()
plt.savefig('Figure2b.svg')
plt.savefig('Figure2b.png')
plt.show()