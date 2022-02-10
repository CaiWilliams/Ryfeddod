from Main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_No_BTM.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 0)
DNG = Dispatch(NG)
StartTime = datetime(year=2016, month=7, day=1, hour=0)
Time = [x for x in range(48)]

MonthNum = 7
ExistingSolar = np.zeros(48)
Si = np.zeros(48)
Scav = np.zeros(48)

for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'Solar':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        ExistingSolar = ExistingSolar + GenN
    if Asset['Technology'] == 'SolarBTM':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        ExistingSolar = ExistingSolar + GenN

#NG = ScalingDynamFromFile(NG, 2, 2, 0, 0,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\100LocationEnhancment.csv')
NG = Scaling(NG,2,0)
DNG = Dispatch(NG)

for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'Solar':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        Si = Si + GenN
    if Asset['Technology'] == 'SolarBTM':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        Si = Si + GenN

#NG = ScalingDynamFromFile(NG, 0, 0, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\100LocationEnhancment.csv')
NG = Scaling(NG,0,1)
DNG = Dispatch(NG)

for Asset in DNG.Distributed.Mix['Technologies']:
    print(Asset['Technology'])
    if Asset['Technology'] == 'SolarNT':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        Scav = Scav + GenN
    if Asset['Technology'] == 'SolarBTMNT':
        Gen = Asset['Generation'].loc[Asset['Generation'].index.month == MonthNum]
        GenN = Gen.groupby([Gen.index.hour, Gen.index.minute]).mean().to_numpy()
        Scav = Scav + GenN

plt.rcParams["figure.dpi"] = 300
FontSize = 14
plt.plot(Time, ExistingSolar,c="tab:green")
plt.plot(Time, Si, c="tab:blue")
plt.plot(Time, Scav, c="tab:orange")
plt.xlim(left=0, right=47)
plt.xticks(range(48)[::12],np.arange(0,48,12)*timedelta(minutes=30),fontsize=FontSize)
plt.xlabel("Time",fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.ylabel("Generation (MW)",fontsize=FontSize)
plt.suptitle('e)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig("Figure2eBangorNoBTM.svg",transparent=True)
plt.savefig("Figure2eBangorNoBTM.png",transparent=True)
plt.show()
