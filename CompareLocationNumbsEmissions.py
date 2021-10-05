import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Main import *
plt.rcParams["figure.dpi"] = 300

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 1, 1)
DNG = Dispatch(NG)
print(1, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\2LocationEnhancment.csv')
DNG = Dispatch(NG)
print(2, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\5LocationEnhancment.csv')
DNG = Dispatch(NG)
print(5, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\10LocationEnhancment.csv')
DNG = Dispatch(NG)
print(10, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\20LocationEnhancment.csv')
DNG = Dispatch(NG)
print(20, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\50LocationEnhancment.csv')
DNG = Dispatch(NG)
print(50, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\100LocationEnhancment.csv')
DNG = Dispatch(NG)
print(100, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\200LocationEnhancment.csv')
DNG = Dispatch(NG)
print(200, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\500LocationEnhancment.csv')
DNG = Dispatch(NG)
print(500, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\1000LocationEnhancment.csv')
DNG = Dispatch(NG)
print(1000, (np.cumsum(DNG.CarbonEmissions) / 2 * (1*10**-9))[-1])
