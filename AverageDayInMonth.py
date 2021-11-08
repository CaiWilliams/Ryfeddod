from Main import *
import matplotlib.pyplot as plt


def AverageDayTechnologiesMonth(NG, Month, **kwargs):
    NG = Grid.Load(NG)
    NG.MatchDates()
    NG.Demand()
    NG.CarbonEmissions()
    #NG.PVGISFetch(kwargs['Device'], kwargs['lat'], kwargs['lon'])
    NG.Modify('Solar',Scaler=1)
    NG.Modify('SolarBTM',Scaler=0)
    NG.Modify('SolarNT', Scaler=0)
    NG.Modify('SolarBTMNT', Scaler=0)
    #DynamScale = NG.DynamScale
    #NG.DynamicScaleingPVGIS('SolarNT', DynamScale, kwargs['SolarNT'])
    #NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, kwargs['SolarBTMNT'])
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
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["figure.dpi"] = 300
    plt.stackplot(range(48), Means)
    plt.xlim(left=0,right=47)
    plt.xticks(range(48)[::8],np.arange(0,48,8)*timedelta(minutes=30),fontsize=12)
    plt.yticks(fontsize=12)
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.xlabel('Time of Day',fontsize=12)
    plt.ylabel('Generation (MW)',fontsize=12)
    plt.tight_layout()
    #plt.savefig('BritainGenerationJuly.png',transparent=True)
    plt.legend(labels)
    return


AverageDayTechnologiesMonth('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM',7)
plt.show()