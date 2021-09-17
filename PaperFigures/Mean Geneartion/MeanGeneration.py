from Main import *
import pandas as pd


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG,1,1,0,0)
DNG = Dispatch(NG)

Month = 7

PVGIS = NG.PVGISData
PVGIS = PVGIS.loc[PVGIS.index.month == Month]
PVGIS = PVGIS[PVGIS['G(i)'] != 0]
print(PVGIS.mean())