from Main import *
import pandas as pd
import numpy as np

Si_Wp = 0.245
Si_Eff = 0.20
DSSC_M = 55
DSSC_Eff = 0.00787
Area_M = 0.13
Capacity_MW = 11912
Capacity_W = Capacity_MW * 1e6

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)

def DSSCperM_toW(Manufacturning, Materials, Efficiency):
    return (Manufacturning*Materials)/(1000*Efficiency)

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (Capacity/1000)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

def CostSweep(Initial_Capacity, A_cost, C_cost, C_start, C_end, C_eff, S_cost ,S_start, S_end, S_eff):
    steps = 20
    C = (np.linspace(C_start,C_end,steps) * Initial_Capacity) + Initial_Capacity
    S = (np.linspace(S_start,S_end,steps) * Initial_Capacity) + Initial_Capacity
    Cost = np.ndarray(shape=(steps, steps))
    Cost.fill(0)

    for idx in range(0,steps,1):
        for jdx in range(0,steps,1):
            Cost[jdx][idx] = (C[jdx] * C_cost) + (S[idx] * S_cost) + (Area(C[jdx], C_eff, 39, 1.968, 7) * A_cost) + (Area(S[idx], S_eff, 39, 1968, 7) * A_cost)
    return Cost

def CarbonSweep(NG, C_start, C_end, S_start, S_end):
    steps = 20
    C = np.linspace(C_start, C_end, steps)
    S = np.linspace(S_start, S_end, steps)
    Carbon = np.ndarray(shape=(steps, steps))
    Carbon.fill(0)

    for idx in range(0,steps,1):
        for jdx in range(0,steps,1):
            NG = Scaling(NG, 1+C[jdx], 1+C[jdx], S[idx], S[idx])
            DNG = Dispatch(NG)
            Carbon[jdx][idx] = np.sum(DNG.CarbonEmissions) / 2 * 1e-9

    Carbon = Carbon[0][0] - Carbon

    return S, C, Carbon

DSSC_Wp = DSSCperM_toW(1.2, DSSC_M, DSSC_Eff)


Cost = CostSweep(Capacity_W, Area_M, Si_Wp, 0, 2, Si_Eff, DSSC_Wp, 0, 2, DSSC_Eff)
X, Y, Carbon = CarbonSweep(NG, 0, 2, 0, 2)
X, Y = np.meshgrid(X, Y)


fig, ax = plt.subplots(dpi=300)
plt.rcParams["figure.figsize"] = (12, 12)
FontSize = 14
surf = ax.pcolormesh(X, Y, Cost/Carbon, cmap='inferno_r', vmax=2e10)

levels = [0.5,1.0,1.5,2,2.5,3,3.5]
cont = ax.contour(X,Y, (X+Y), levels=levels, colors="w")
ax.clabel(cont, fontsize=10, inline=True)

plt.xlabel("C$_s$", fontsize=FontSize)
plt.xticks(fontsize=FontSize)
cbar = plt.colorbar(surf)
cbar.set_label(label='CO$_2$e Emissions Saved per Unit Cost (Mt Saved/\$)', size=12)
cbar.ax.tick_params(labelsize=12)
plt.ylabel('C$_c$', fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.suptitle('b)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
#plt.savefig("Figure3b.svg")
#plt.savefig("Figure3b.png")
plt.show()