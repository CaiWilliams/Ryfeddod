from Main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM')
device_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
Irradiance = NG.PVGISData.to_numpy()
Irradiance = Irradiance[Irradiance != 0]
plt.rcParams["figure.dpi"] = 300
FontSize = 14
plt.hist(Irradiance, bins=30)
plt.ylabel("Frequency", fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.xlabel("Irradiance (Wm$^{-2}$)", fontsize=FontSize)
plt.yticks(fontsize=FontSize)
plt.xlim(left=0)
plt.suptitle('c)',x=0.05,y=0.99,fontsize=FontSize)
plt.tight_layout()
plt.savefig('Figure2d_30.svg',transparent=True)
plt.savefig('Figure2d_30.png',transparent=True)
#plt.show()