from Main import *
import pandas as pd
import numpy as np

Si_Wp = 0.245
Si_Eff = 0.2
DSSC_M = 65.09
DSSC_Eff = 0.0722
Area_M = 0.13
Capacity_MW = 11912
Capacity_W = Capacity_MW * 1e6


fig, ax = plt.subplots(dpi=300)
NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle0M.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)

def DSSCperM_toW(Manufacturning, Materials, Efficiency):
    return (Manufacturning*Materials)/(1000*Efficiency)

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (Capacity/1000)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

def Area_to_Capacity(Area, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    PerPanArea = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2)
    Cap = (Area/PerPanArea) * 1000 * Eff
    return Cap

def CarbonSweep(NG,C,S):
    steps = len(C)
    Carbon = np.ndarray(shape=(steps, steps))
    Carbon.fill(0)

    for idx in range(0,steps,1):
        for jdx in range(0,steps,1):
            NG = Scaling(NG, C[jdx], C[jdx], S[idx], S[idx])
            DNG = Dispatch(NG)
            Carbon[jdx][idx] = np.sum(DNG.CarbonEmissions) / 2 * 1e-9

    Carbon = Carbon[0][0] - Carbon
    return Carbon


X = np.linspace(0,2,100)
Y = np.linspace(0,2,100)
Xm, Ym, = np.meshgrid(X,Y)

Initial_Area = Area(Capacity_W, Si_Eff, 39, 1.968, 7)

Si_Capacity = [Area_to_Capacity((Initial_Area*x)+Initial_Area, Si_Eff, 39, 1.968, 7) for x in X]
DSSC_Capacity = [Area_to_Capacity((Initial_Area*x)+Initial_Area, DSSC_Eff, 39, 1.968, 7) for x in X]

Si_Scaler = [x/Capacity_W for x in Si_Capacity]
DSSC_Scaler = [x/Capacity_W for x in DSSC_Capacity]
print(Si_Scaler)
print(DSSC_Scaler)
Carbon = CarbonSweep(NG, Si_Scaler, DSSC_Scaler)
X_Area = X * Initial_Area
Y_Area = Y * Initial_Area
X_Area, Y_Area = np.meshgrid(X_Area, Y_Area)

fig, ax = plt.subplots(dpi=300)
FontSize = 14
surf = ax.pcolor(Xm, Ym, Carbon/(X_Area+Y_Area), cmap='inferno', vmax=4e-8, vmin=0.5e-8)

levels = [0.5,1.0,1.5,2,2.5,3,3.5]
cont = ax.contour(Xm,Ym, (Xm+Ym), levels=levels, colors="w")
ax.clabel(cont, fontsize=10, inline=True)

plt.ylabel("A$_C$", fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.xlabel("A$_S$", fontsize=FontSize)
plt.xticks(fontsize=FontSize)
cbar = plt.colorbar(surf)
cbar.set_label(label='CO$_2$e Emissions Saved per Unit Area (Mt Saved/m$^2$)', size=12)
cbar.ax.tick_params(labelsize=12)
#plt.suptitle('c)',x=0.05,y=0.99,fontsize=FontSize)
plt.savefig("Figure3cNewcastle0MCC.svg")
plt.savefig("Figure3cNewcastle0MCC.png")
plt.show()