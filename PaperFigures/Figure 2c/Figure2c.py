from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def SweepCarbon(NG,C_Start,C_End,S_Start,S_End):

    steps = 100
    Conv = np.linspace(C_Start,C_End, steps)
    Scav = np.linspace(S_Start,S_End, steps)

    Gen = np.ndarray(shape=(steps, steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        for jdx in range(0,steps,1):
            NG = Scaling(NG, Conv[jdx]+1, Conv[jdx]+1, Scav[idx], Scav[idx])
            DNG = Dispatch(NG)

            Gen[jdx][idx] = (np.sum(DNG.CarbonEmissions) / 2 * (1*10**-9))
    Gen = Gen[0][0] - Gen
    return Conv, Scav, Gen

fig, ax = plt.subplots(dpi=300)
NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
C, S, Gen = SweepCarbon(NG,0,2,0,2)
C, S = np.meshgrid(C,S)
surf = ax.pcolor(C,S,Gen,cmap='inferno')

levels = [0.5,1.0,1.5,2,2.5,3,3.5]
cont = ax.contour(C,S, (C+S), levels=levels, colors="w")
ax.clabel(cont, fontsize=10, inline=True)


FontSize=14
plt.ylim(bottom=0)
plt.xlabel("C$_s$",fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.ylabel("C$_c$",fontsize=FontSize)
plt.yticks(fontsize=FontSize)
cbar = plt.colorbar(surf)
cbar.set_label(label="CO$_2$e Emissions Savings (Mt)", size=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
plt.tight_layout()
plt.savefig("CarbonSavings.svg")
plt.savefig("CarbonSavings.png")
plt.show()