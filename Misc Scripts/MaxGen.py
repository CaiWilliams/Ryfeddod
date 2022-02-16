import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
import pytz
import requests
import io
from pvlive_api import PVLive
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy


def Enh(EnhancmentDir):
    Enhancment = pd.read_csv(EnhancmentDir)
    f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
    return f

Irr = np.linspace(0,1000,100000)
f = Enh('Data/Devices/DSSC.csv')
Pow = f(Irr)
#plt.plot(Irr,Pow)

#f2 = Enh('Data/Devices/NewCastle.csv')
#Pow2 = f2(Irr)
#plt.plot(Irr,Pow2)

Eq = 1000/Irr
plt.plot(Irr,Pow-Eq)
#Eq = 1000/Irr/2
#plt.plot(Irr,Eq)

plt.ylabel("Enhancement")
plt.xlabel("Irradiance ($Wm^{-2}$)")
#plt.ylim(0,np.max(Pow)+0.5)
plt.show()