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
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy





EnhancmentDir = 'Data/Devices/test.csv'
y1 = np.arange(0,1000,1)
y2 = np.arange(1000,0,-0.01)
y = np.concatenate((y1,y2))
x = range(len(y1))

Enhancment = pd.read_csv(EnhancmentDir)
f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
DynamScale = f(y1)
plt.plot(x,DynamScale)
plt.show()