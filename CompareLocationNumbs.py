import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Main import *
plt.rcParams["figure.dpi"] = 300

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 1, 1)
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="1")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\2LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="2")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\5LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="5")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\10LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="10")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\20LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="20")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\50LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="50")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\100LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="100")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\200LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="200")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\500LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="500")

NG = Setup('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\\2016RawT.NGM', 'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Devices\DSSC.csv', 53.13359, -1.746826)
NG = Scaling(NG, 1, 1, 0, 0)
NG = ScalingDynamFromFile(NG, 1, 1, 1, 1,'C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\\1000LocationEnhancment.csv')
DNG = Dispatch(NG)
for Asset in DNG.Distributed.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        plt.plot(np.cumsum(Asset['Generation']/ 1000000 / 2), label="1000")
plt.ylabel("Energy Generated (TWh)")
plt.legend()
plt.show()