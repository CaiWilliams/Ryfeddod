from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

grid_dir = os.path.join(os.path.dirname(os.getcwd()),'Data','NationalGrid_2016.NGM')
device_dir = os.path.join(os.path.dirname(os.getcwd()),'Data','Devices','DSSC.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
DNG = Dispatch(NG)
DNG = DNG.run(1,0)

print("Cc = 0 Cs = 0, Bangor")
label = 'Nuclear'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Nuclear_Generation = Assets['Generation'].values
        Nuclear_Capacity = Assets['Capacity']

Max_Nuclear_Gen = 365 * 24 * Nuclear_Capacity
Nuclear_Generation = np.sum(Nuclear_Generation/2)
Nuclear_Capacity_Factor = (Nuclear_Generation / Max_Nuclear_Gen) * 100
#rint("Nuclear Capacity Factor: " + str(Nuclear_Capacity_Factor.round(3)) + " %")


label = 'Solar'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']
    OG_Carbon = np.sum(Assets['CarbonEmissions'] / 2 )

Max_Solar_Gen = 365 * 24 * Solar_Capacity
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("Solar Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")

device_dir = os.path.join(os.path.dirname(os.getcwd()),'Data','Devices','DSSC.csv')
NG = setup(grid_dir, device_dir, 53.13359, -1.746826)
DNG = Dispatch(NG)
DNG = DNG.run(1,1)

print("Cc = 0 Cs = 1, Bangor")
label = 'Nuclear'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Nuclear_Generation = Assets['Generation'].values
        Nuclear_Capacity = Assets['Capacity']

Max_Nuclear_Gen = 365 * 24 * Nuclear_Capacity
Nuclear_Generation = np.sum(Nuclear_Generation/2)
Nuclear_Capacity_Factor = (Nuclear_Generation / Max_Nuclear_Gen) * 100
#print("Nuclear Capacity Factor: " + str(Nuclear_Capacity_Factor.round(3)) + " %")

label = 'Solar'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Max_Solar_Gen = 365 * 24 * Solar_Capacity
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("Solar Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")


label = 'SolarNT'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Max_Solar_Gen = 365 * 24 * Solar_Capacity
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("SolarNT Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")
