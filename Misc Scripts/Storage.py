from Main import *
import matplotlib.pyplot as plt



def StorageUsage(NG,SNTS):
    Steps = 2
    SNTS = np.linspace(0, SNTS, Steps)
    U = np.zeros(Steps)
    for idx,x in enumerate(SNTS):
        DNG = Dispatch(NG, 1, 1)
        #for Asset in DNG.Distributed.Mix['Technologies']:
            #if Asset['Technology'] == 'Hydro Pumped Storage':
            #   U[idx] = np.sum(Asset['Generation']/ 1000000 / 2)

        plt.plot(DNG.StorageSOC)
        #plt.plot(DNG.StorageDischarge)
        #plt.plot(DNG.StorageDemandRemaining)
        #p#lt.plot(DNG.DC2Curtailed)
    return U,SNTS


NG = setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016BMRS.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
U,SNTS = StorageUsage(NG,2)
#plt.plot(SNTS,U)
#plt.plot(SNTS,Y)
#plt.ylabel('Energy Generated (TWh)')
#plt.xlabel('C')
plt.show()