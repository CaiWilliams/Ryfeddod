from Main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NG = setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\NationalGrid_2016.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
DNG = Dispatch(NG,1,0)
StartTime = datetime(year=2016, month=7, day=1, hour=0)
Time = [x for x in range(48)]

MonthNum = 7
ExistingSolar = np.zeros(48)
Scav = np.zeros(48)

for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'Solar':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        ExistingSolar = ExistingSolar + GenN


DNG = Dispatch(NG, 1, 1)

for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        Scav = Scav + GenN


plt.rcParams["figure.dpi"] = 300
plt.plot(Time, ExistingSolar, c="y")
plt.plot(Time, Scav, c="m")
plt.xlim(left=0, right=47)
plt.xticks(range(48)[::8],np.arange(0,48,8)*timedelta(minutes=30))
plt.xlabel("Time")
plt.ylabel("Generation (MW)")
plt.savefig("Figure2f.svg")
plt.savefig("Figure2f.png")
#plt.show()
