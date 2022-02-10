from Main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_No_BTM.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG,1,0)
DNG = Dispatch(NG)

print("Cc = 0 Cs = 0, Bangor")
label = 'Nuclear'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Nuclear_Generation = Assets['Generation'].values
        Nuclear_Capacity = Assets['Capacity']

Max_Nuclear_Gen = 365 * 24 * Nuclear_Capacity
Nuclear_Generation = np.sum(Nuclear_Generation/2)
Nuclear_Capacity_Factor = (Nuclear_Generation / Max_Nuclear_Gen) * 100
print("Nuclear Capacity Factor: " + str(Nuclear_Capacity_Factor.round(3)) + " %")


label = 'Solar'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Max_Solar_Gen = 365 * 24 * 11970
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("Solar Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")


NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016_No_BTM.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\\Newcastle48U.csv', 53.13359, -1.746826)
NG = Scaling(NG,0,1)
DNG = Dispatch(NG)

print("Cc = 1 Cs = 1, Bangor")
label = 'Nuclear'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Nuclear_Generation = Assets['Generation'].values
        Nuclear_Capacity = Assets['Capacity']

Max_Nuclear_Gen = 365 * 24 * Nuclear_Capacity
Nuclear_Generation = np.sum(Nuclear_Generation/2)
Nuclear_Capacity_Factor = (Nuclear_Generation / Max_Nuclear_Gen) * 100
print("Nuclear Capacity Factor: " + str(Nuclear_Capacity_Factor.round(3)) + " %")

label = 'Solar'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Max_Solar_Gen = 365 * 24 * 11970
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("Solar Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")


label = 'SolarNT'
for Assets in DNG.Distributed.Mix['Technologies']:
    if Assets['Technology'] == label:
        Solar_Generation = Assets['Generation'].values
        Solar_Capacity = Assets['Capacity']

Max_Solar_Gen = 365 * 24 * 11970
Solar_Generation = np.sum(Solar_Generation/2)
Solar_Capacity_Factor = (Solar_Generation / Max_Solar_Gen) * 100
print("SolarNT Capacity Factor: " + str(Solar_Capacity_Factor.round(3)) + " %")
