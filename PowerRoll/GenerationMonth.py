from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AverageDayTechnologiesMonthA(NG, Month):
    DNG = Dispatch(NG)
    Means = np.ndarray(shape=(len(DNG.Distributed.Mix['Technologies']),48))
    Means.fill(0)
    #for idx,Technology in enumerate(args):
    for idx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
        if Asset['Technology'] == 'Nuclear':
            N = Asset['Generation'].loc[Asset['Generation'].index.month == Month]
            N = np.sum(N)
            print(N)
        M = Asset['Generation'].loc[Asset['Generation'].index.month == Month]
        Means[idx] = M.groupby([M.index.hour,M.index.minute]).mean().to_numpy()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    idx = [1, 0, 10, 10, 10, 10, 2, 3, 4, 5, 7, 9, 6, 8, 10]
    plt.stackplot(range(48), Means[idx])
    plt.xlim(left=0,right=47)
    plt.xticks(range(48)[::8],np.arange(0,48,8)*timedelta(minutes=30))
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    print(labels)
    plt.xlabel('Time of Day',fontsize=14)
    plt.ylabel('Generation (MW)',fontsize=14)
    #plt.tight_layout()
    #plt.legend(labels)
    return

#plt.text(0.1,10000,'Fossil Gas',c='white')

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG,1,1,0,0)
plt.rcParams["figure.figsize"] = (6.5, 6)
plt.rcParams["figure.dpi"] = 300
AverageDayTechnologiesMonthA(NG,5)
FontSize = 14
plt.xticks(fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.text(2,5000,'Fossil Gas',c='white')
plt.annotate('Fossil Hard Coal', xy=(3,100),xycoords='data', xytext=(2,3500),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.text(2,16000,'Nuclear', c='white')

plt.annotate('SolarBTM', xy=(24,45000),xycoords='data', xytext=(25,13000),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Solar', xy=(23,40000),xycoords='data', xytext=(25,11500),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Wind Onshore', xy=(22,30000),xycoords='data', xytext=(25,10000),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Wind Offshore', xy=(21,27500),xycoords='data', xytext=(25,8500),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Run-of-River', xy=(20,25700),xycoords='data', xytext=(25,7000),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.annotate('Hydro Pumped Storage', xy=(19,25000),xycoords='data', xytext=(25,5500),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10",color="white"),c='white')
plt.tight_layout()
plt.savefig('GenerationMonth2.png')
plt.show()