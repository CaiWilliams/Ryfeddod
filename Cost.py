import copy

import numpy as np
import matplotlib.pyplot as plt


def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (1000 * Capacity)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

def ModuleCost(Capacity, ModCost):
    return Capacity * ModCost

Carbon = np.genfromtxt('Misc Data/resutlsCBangor22.csv', delimiter=',')
Carbon = np.nan_to_num(Carbon)

Cost = np.genfromtxt('Misc Data/resutlsSBangor22.csv', delimiter=',')
Cost = np.nan_to_num(Cost)
r = Cost
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig, ax = plt.subplots(dpi=300)


X = np.arange(0, 2.05, 0.05)
Y = np.arange(1, 2.05, 0.05)
x = X
y = Y

AreaCost = 0.18532887226
DSSC_Cost = 9.87179
DSSC_Eff = 0.0078
Si_Cost = 0.18
Si_Eff = 0.15
Capacity = 11798.3 * 1e6

print()
Xa = Area(Capacity * X, DSSC_Eff, 39, 1.968, 7) * AreaCost
Ya = Area(Capacity * Y, Si_Eff, 39, 1.968, 7) * AreaCost
Xm = X * Capacity * Si_Cost
Ym = Y * Capacity * DSSC_Cost
Xc = Xa + Xm
Yc = Ya + Ym
xc = Xc
yc = Yc
Xc, Yc = np.meshgrid(Xc, Yc)
X, Y = np.meshgrid(X, Y)
#Now = np.where(y == 1)[0][0]

ra = Area(Capacity * r, Si_Eff, 39, 1.968, 7) * AreaCost
rm = r * Capacity * Si_Cost
rc = rm + ra
#print(Xc+Yc)


#Carbon = np.where((Carbon[Now][0]-Carbon) < 0,np.nan,Carbon)
#Carbon = Carbon[0][0] - Carbon
#Z = Carbon
surf = ax.pcolor(X, Y, (Xc+Yc)/rc)

levels = [1.5, 2, 2.5, 3, 3.5]
CS = ax.contour(X, Y, (X+Y), levels=levels, colors='w')
ax.clabel(CS, fontsize=9, inline=True)

plt.ylabel("Silicon Capacity Scaler")
plt.xlabel("Emerging PV Capacity Scaler")
#surf.set_clim(0,5e7)
fig.colorbar(surf,label="Scaler for an exclusively silicon grid with equal CO$_2$e emissions")
plt.show()