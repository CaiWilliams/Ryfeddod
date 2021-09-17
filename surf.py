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

s = np.genfromtxt('Misc Data/resutlsSNewCastle2.csv', delimiter=',')
s = np.nan_to_num(s)
t = np.genfromtxt('Misc Data/resutlsTNewCastle2.csv', delimiter=',')
t = np.nan_to_num(t)

r = np.genfromtxt('Misc Data/resultsHCSCoal.csv', delimiter=',')
r = np.nan_to_num(r)

r2 = np.genfromtxt('Misc Data/resultsHCSGas.csv', delimiter=',')
r2 = np.nan_to_num(r2)

r3 = np.genfromtxt('Misc Data/resultsHCSSCoal.csv', delimiter=',')
r3 = np.nan_to_num(r3)

r4 = np.genfromtxt('Misc Data/resultsHCSSGas.csv', delimiter=',')
r4 = np.nan_to_num(r4)

dc3 = np.genfromtxt('Misc Data/resultsHCSDS3.csv', delimiter=',')
dc3s = np.genfromtxt('Misc Data/resultsHCSSDS3.csv', delimiter=',')

C = np.genfromtxt('Misc Data/resutlsCNewCastle2.csv', delimiter=',')
C = np.nan_to_num(C)

solarG = np.genfromtxt('Misc Data/resultsHCSSolarG.csv', delimiter=',')
solarNTG = np.genfromtxt('Misc Data/resultsHCSSolarNTG.csv', delimiter=',')
solarG = solarG + solarNTG
solarSG = np.genfromtxt('Misc Data/resultsHCSSSolarG.csv', delimiter=',')

solar = np.genfromtxt('Misc Data/resultsHCSSolar.csv', delimiter=',')
solarNT = np.genfromtxt('Misc Data/resultsHCSSolarNT.csv', delimiter=',')
solar = solar + solarNT
solarS = np.genfromtxt('Misc Data/resultsHCSSSolar.csv', delimiter=',')

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
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
#Ya = Area(Capacity, Si_Eff, 39, 1.968, 7) * AreaCost
#Xm = Capacity * Si_Cost
#Ym = Capacity * DSSC_Cost
#C0 = Ya + Xm


#Xa = Area(Capacity * X, DSSC_Eff, 39, 1.968, 7) * AreaCost
#Ya = Area(Capacity * Y, Si_Eff, 39, 1.968, 7) * AreaCost
#Xm = X * Capacity * Si_Cost
#Ym = Y * Capacity * DSSC_Cost
#Xc = Xa + Xm
#Yc = Ya + Xm
#Xc, Yc = np.meshgrid(Xc, Yc)

#Xc, Yc = np.meshgrid(Xc, Yc)
X, Y = np.meshgrid(X, Y)


#ra = Area(Capacity * r, Si_Eff, 39, 1.968, 7) * AreaCost
#rm = r * Capacity * Si_Cost
#rc = rm + ra

#C = (Area(Capacity, Si_Eff, 39, 1.968, 7) * AreaCost) + (Capacity * Si_Cost)

#a = (Xc+Yc)/C
#rc = rc/C
#print(Xc[-1][-1] + Yc[-1][-1])
#print(rc[-1][-1])
#rcc = (rc)/(Xc+Yc)
a = r + r2
a0 = a[0][0]
a = a[0][0] - a
b = r3 + r4
b[0][0] = a0
b = b[0][0] - b
#c = b/a
#c[0][0] = 1



surf = ax.pcolor(X, Y, solarSG, cmap='inferno')
plt.xlim(left=0)
plt.ylim(bottom=0)
#line = ax.scatter(r_y[:-1], r_x[:-1], r_r[:-1], alpha=1, color='r')
plt.ylabel("Silicon Capacity Scaler")
plt.xlabel("Emerging PV Capacity Scaler")
fig.colorbar(surf,label="DC4 Generation axes mix / DC4 Generation silicon only")
plt.show()



