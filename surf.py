import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (1000 * Capacity)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

def ModuleCost(Capacity, ModCost):
    return Capacity * ModCost


r = np.genfromtxt('resutlsSNewCastle2.csv',delimiter=',')
r = np.nan_to_num(r)

C = np.genfromtxt('resutlsCBangor2.csv',delimiter=',')
C = np.nan_to_num(C)

fig, ax = plt.subplots()

X = np.arange(0,2,0.05)
Y = np.arange(0,2,0.05)


AreaCost = 0.18532887226
DSSC_Cost = 0.78571 * 1e6
DSSC_Eff = 0.098
Si_Cost = 0.18 * 1000
Si_Eff = 0.15
Capacity = 11798.3

#Xa = Area(Capacity, DSSC_Eff, 39/, 1.968, 7) * AreaCost
Ya = Area(Capacity, Si_Eff, 39, 1.968, 7) * AreaCost
Xm = Capacity * Si_Cost
Ym = Capacity * DSSC_Cost
C0 = Ya + Xm


Xa = Area(Capacity * X, DSSC_Eff, 39, 1.968, 7) * AreaCost
Ya = Area(Capacity * Y, Si_Eff, 39, 1.968, 7) * AreaCost
Xm = X * Capacity * Si_Cost
Ym = Y * Capacity * DSSC_Cost
Xc = Xa + Xm
Yc = Ya + Xm
Xc, Yc = np.meshgrid(Xc, Yc)

#Xc, Yc = np.meshgrid(Xc, Yc)
X, Y = np.meshgrid(X, Y)


ra = Area(Capacity * r, Si_Eff, 39, 1.968, 7) * AreaCost
rm = r * Capacity * Si_Cost
rc = rm + ra

#C = (Area(Capacity, Si_Eff, 39, 1.968, 7) * AreaCost) + (Capacity * Si_Cost)

#a = (Xc+Yc)/C
#rc = rc/C
print(Xc[-1][-1] + Yc[-1][-1])
print(rc[-1][-1])
rcc = (rc)/(Xc+Yc)


surf = ax.pcolor(X, Y, rcc, cmap='inferno')
plt.xlim(left=0)
plt.ylim(bottom=0)
#line = ax.scatter(r_y[:-1], r_x[:-1], r_r[:-1], alpha=1, color='r')
plt.ylabel("Silicon Capacity Scaler")
plt.xlabel("Emerging PV Capacity Scaler")
fig.colorbar(surf,label="X times more/less expensive to use only Silicon")
plt.show()



