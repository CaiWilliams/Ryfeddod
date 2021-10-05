from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (Capacity/1000)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG,1,1,0,0)
DNG = Dispatch(NG)

Irr = NG.PVGISData.to_numpy().ravel()
Solar_label = 'Solar'

for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == Solar_label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Input = Irr * Area(Solar_Capacity * 1e6, 0.20, 39, 1.968, 7)
Solar_Generation = Solar_Generation * 1e6
plt.scatter(Irr, (Solar_Generation/Input)*100)
plt.xlim(left=0)
#plt.ylim(top=30)
#plt.xscale('symlog')
plt.show()