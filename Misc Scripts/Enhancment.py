
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
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


def Plot_Enhancment(EnhancmentDir):
    X = np.arange(0,1000,0.01)
    Enhancment = pd.read_csv(EnhancmentDir)
    f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
    A = f(X)
    plt.plot(X,A)
    return

Bangor = 'Data/Devices/DSSC.csv'
Newcastle48U = 'Data/Devices/Newcastle48U.csv'
Newcastle24U = 'Data/Devices/Newcastle24U.csv'
Newcastle18U = 'Data/Devices/Newcastle18U.csv'
Newcastle12U = 'Data/Devices/Newcastle12U.csv'


Plot_Enhancment(Bangor)
Plot_Enhancment(Newcastle12U)
Plot_Enhancment(Newcastle18U)
Plot_Enhancment(Newcastle24U)
Plot_Enhancment(Newcastle48U)
#X = np.arange(0,1000,0.01)
#Y = 1000/X
#plt.plot(X,Y)
plt.xlabel('Irradiance (Wm$^{-2}$)')
plt.ylabel('Enhancment')
#plt.xlim(left=0, right=1000)
#plt.ylim(bottom=0, top=15 )
plt.tight_layout()
plt.show()