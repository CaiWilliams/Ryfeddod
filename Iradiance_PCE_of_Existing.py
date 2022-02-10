from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (Capacity/1000)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_No_BTM.NGM','C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 0)
DNG = Dispatch(NG)

Irr = NG.PVGISData.to_numpy().ravel()
Solar_label = 'Solar'

for Assets in NG.Mix['Technologies']:
    if Assets['Technology'] == Solar_label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']  # / 1e3
        Scaler = Assets['Scaler']
data = pd.DataFrame()
data['Gen'] = Solar_Generation
data['Irr'] = Irr
data['Irr'] = data['Irr']
data = data.groupby('Irr').mean().reset_index()
plt.scatter(data['Irr'], data['Gen'])







#plt.plot(np.ones(1200) * Solar_Capacity, c='tab:green')
#plt.plot(np.ones(1200) * 11654, c='tab:red')
#plt.plot(np.ones(1200) * 11970, c='tab:purple')
plt.xlabel('Irradiance (Wm^-2)')
plt.ylabel('Generation (MW)')
#plt.xlim(left=0, right=1000)
plt.show()
