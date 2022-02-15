from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def SweepCarbon(NG,C_Start,C_End,S_Start,S_End):

    steps = 51
    Conv = np.linspace(C_Start,C_End, steps)
    Scav = np.linspace(S_Start,S_End, steps)

    Gen = np.ndarray(shape=(steps, steps))
    Gen.fill(0)

    for idx in range(0,steps,1):
        for jdx in range(0,steps,1):
            DNG = Dispatch(NG,Conv[jdx]+1, Scav[idx])

            Gen[jdx][idx] = (np.sum(DNG.CarbonEmissions) / 2 * (1*10**-9))
    Gen = Gen[0][0] - Gen
    return Conv, Scav, Gen

fig, ax = plt.subplots(dpi=300)
NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_CorrectCapacity.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
C, S, Gen = SweepCarbon(NG,0,2,0,2)
C, S = np.meshgrid(C,S)
surf = ax.pcolor(C,S,Gen,cmap='inferno')

levels = [5,10,15,20,25,30,35,40,45,50,55,60]
cont = ax.contour(C,S, Gen, levels=levels, colors="w")
ax.clabel(cont, fontsize=10, inline=True)


FontSize = 14
plt.ylim(bottom=0)
plt.xlabel("C$_s$",fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.ylabel("C$_c$",fontsize=FontSize)
plt.yticks(fontsize=FontSize)
cbar = plt.colorbar(surf)
cbar.set_label(label="CO$_2$e Emissions Savings (Mt)", size=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
plt.suptitle('d)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
#plt.savefig("CarbonSavingsBangorNoBTMNewcastle48U.svg",transparent=True)
#plt.savefig("CarbonSavingsBangorNoBTMNewcastle48U.png",transparent=True)
plt.show()