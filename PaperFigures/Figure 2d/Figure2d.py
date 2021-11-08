from Main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
Irradiance = NG.PVGISData.to_numpy()
Irradiance = Irradiance[Irradiance != 0]
plt.rcParams["figure.dpi"] = 300
FontSize = 14
plt.hist(Irradiance, bins=50)
plt.ylabel("Frequency", fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.xlabel("Irradiance (Wm$^{-2}$)", fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.xlim(left=0)
plt.suptitle('d)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('Figure2d_50.svg')
plt.savefig('Figure2d_50.png')
plt.show()