from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Si_Wp = 0.245
Si_Eff = 0.2
DSSC_M = 65.09
DSSC_Eff = 0.0989
Area_M = 0.13
Capacity_MW = 11912
Capacity_W = Capacity_MW * 1e6

def DSSCperM_toW(Manufacturning, Materials, Efficiency):
    return (Manufacturning*Materials)/(1000*Efficiency)

def Area(Capacity, Eff, Tilt, Width, Elev):
    RowWidth = Elev + (np.cos(np.radians(Tilt)) * Width)
    NofPanels = (1000 * Capacity)/Eff
    Area = (((((1.92 * np.cos(np.radians(Tilt))) * 2) + RowWidth) * 0.99)/2) * NofPanels
    return Area

DSSC_Wp_Low = DSSCperM_toW(1.2, DSSC_M, DSSC_Eff)
DSSC_Wp_Mean = DSSCperM_toW(1.3, DSSC_M, DSSC_Eff)
DSSC_Wp_High = DSSCperM_toW(1.4, DSSC_M, DSSC_Eff)

X = np.linspace(0,2,1000)

Initial_Cost = (Capacity_W * Si_Wp) + (Area(Capacity_MW * 1, Si_Eff, 39, 1.968, 7) * Area_M)

Si_Cost = (X * Capacity_W * Si_Wp) + (Area(Capacity_MW * X, Si_Eff, 39, 1.968, 7) * Area_M)
DSSC_Cost_Low = (X * Capacity_W * DSSC_Wp_Low) + (Area(Capacity_MW * X, DSSC_Eff, 39, 1.968, 7) * Area_M)
DSSC_Cost_Mean = (X * Capacity_W * DSSC_Wp_Mean) + (Area(Capacity_MW * X, DSSC_Eff, 39, 1.968, 7) * Area_M)
DSSC_Cost_High = (X * Capacity_W * DSSC_Wp_High) + (Area(Capacity_MW * X, DSSC_Eff, 39, 1.968, 7) * Area_M)

plt.rcParams["figure.dpi"] = 300
FontSize = 14
plt.plot(X, Si_Cost+Initial_Cost, c="tab:blue")
plt.plot(X, DSSC_Cost_Low+Initial_Cost, c="tab:orange", linestyle="--")
plt.plot(X, DSSC_Cost_Mean+Initial_Cost, c="tab:orange")
plt.plot(X, DSSC_Cost_High+Initial_Cost, c="tab:orange", linestyle="--")
plt.xlabel("C", fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.ylabel("Cost of Installed Capacity ($)", fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.suptitle('a)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig("Figure3a.svg")
plt.savefig("Figure3a.png")
#plt.show()