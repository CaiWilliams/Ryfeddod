from Main import *
import matplotlib.pyplot as plt

NG = setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\NationalGrid_2016.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
DNG = Dispatch(NG, 1, 0)
plt.plot(np.array(DNG.StorageDischarge))

for Asset in NG.Mix['Technologies']:
    if Asset['Technology'] == 'Hydro Pumped Storage':
        print(Asset['Scaler'])
        plt.plot(np.array(Asset['Generation']))

plt.show()