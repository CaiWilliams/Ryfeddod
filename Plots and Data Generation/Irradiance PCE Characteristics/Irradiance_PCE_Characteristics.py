import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

def linfit(x):
    mx + c


FontSize = 14
plt.rcParams["figure.dpi"] = 300
Bangor_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSCPCE.csv')
Bangor = pd.read_csv(Bangor_dir)

u48_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle48UPCE.csv')
u24_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle24UPCE.csv')
u18_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle18UPCE.csv')
u12_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','Newcastle12UPCE.csv')

u48 = pd.read_csv(u48_dir)
u24 = pd.read_csv(u24_dir)
u18 = pd.read_csv(u18_dir)
u12 = pd.read_csv(u12_dir)

plt.scatter(u48['Irradiance'],u48['Enhanced'],c='#6a51a3')
plt.scatter(u24['Irradiance'],u24['Enhanced'],c='#807dba')
plt.scatter(u18['Irradiance'],u18['Enhanced'],c='#9e9ac8')
plt.scatter(u12['Irradiance'],u12['Enhanced'],c='#bcbddc')

x = np.linspace(140,1000,100000)
f = np.interp(x,u48['Irradiance'], u48['Enhanced'])
plt.plot(x, f, c='#6a51a3',linestyle='--')

f = np.interp(x,u24['Irradiance'], u24['Enhanced'])
plt.plot(x, f, c='#807dba',linestyle='--')

f = np.interp(x,u18['Irradiance'], u18['Enhanced'])
plt.plot(x, f, c='#9e9ac8',linestyle='--')

f = np.interp(x,u12['Irradiance'], u12['Enhanced'])
plt.plot(x, f, c='#bcbddc',linestyle='--')

plt.plot(Bangor['Irradiance'][70:],Bangor['Enhanced'][70:]*100,c='#4a1486')
plt.ylim(bottom=0)
#plt.xlim(left=0, right=1000)
plt.ylabel('Power Conversion Efficiency (%)', fontsize=FontSize)
plt.xticks(fontsize=FontSize)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10],fontsize=FontSize)
plt.xlabel('Irradiance (Wm$^{-2}$)',fontsize=FontSize)
plt.tight_layout()
plt.savefig("Figure1Transparent.png",transparent=True)
plt.savefig("Figure1Transparent.svg",Transparent=True)
plt.show()