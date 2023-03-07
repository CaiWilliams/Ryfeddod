import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from Main_old import *
plt.rcParams["figure.dpi"] = 300


x = [1,2,5,10,20,50,100,200,500,1000]
y = np.zeros(len(x))
NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[0] = np.sum(Asset['Generation']/ 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="1")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','2LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[1] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="2")
#
NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','5LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[2] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="5")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','10LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[3] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="10")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','20LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[4] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="20")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','50LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[5] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="50")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','100LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[6] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="100")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','200LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[7] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="200")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','500LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[8] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="500")

NG = Setup(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','NationalGrid_2016.NGM'), os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC.csv'), 53.13359, -1.746826)
NG = Scaling(NG, 1, 1)
NG = ScalingDynamFromFile(NG, 1, 1,os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Data','Devices','DSSC Multi Location','1000LocationEnhancment.csv'))
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarNT':
        y[9] = np.sum(Asset['Generation'] / 1000000 / 2)
        #plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="1000")
plt.plot(x,y)
plt.xscale('log')
plt.ylabel("Energy Generated (TWh)")
plt.xlabel("Locations Used")
plt.savefig('LocationsUsed.png')
