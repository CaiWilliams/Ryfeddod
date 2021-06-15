import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from Dispatch import *
from Plots import *

#UK = Original('Data/2015/UK','UK')
UKSolar = Original('Data/2015/UKSolar','UKSolar')
UK2 = Original('Data/2015/UK2', 'UK2')
EM = Original('Data/2015/Enhancment', 'Enhancment')

UKSolar = UKSolar.MatchTimes(UK2)
UK2 = UK2.MatchTimes(UKSolar)
EM = EM.MatchTimes(UK2)
EM = EM.MatchTimes(UKSolar)
UKSolar = UKSolar.MatchTimes(EM)
UK2 = UK2.MatchTimes(EM)
EMa = np.ravel(EM.Data['Enhancment'].to_numpy())


Gas = Technology(UK2, 'Gas')
Gas.SetCapacity(38274)
Gas.SetCarbonIntensity(443)
Gas.SetScaler(1)

Coal = Technology(UK2, 'Coal')
Coal.SetCapacity(6780)
Coal.SetCarbonIntensity(960)
Coal.SetScaler(1)

PumpedStorage = Technology(UK2, 'PumpedStorage')
PumpedStorage.SetCapacity(4052)
PumpedStorage.SetCarbonIntensity(12)
PumpedStorage.SetScaler(1)

Hydro = Technology(UK2, 'Hydro')
Hydro.SetCapacity(1882)
Hydro.SetCarbonIntensity(10)
Hydro.SetScaler(1)

Nuclear = Technology(UK2, 'Nuclear')
Nuclear.SetCapacity(8209)
Nuclear.SetCarbonIntensity(13)
Nuclear.SetScaler(1)

WindOffshore = Technology(UK2, 'WindOffshore')
WindOffshore.SetCapacity(10365)
WindOffshore.SetCarbonIntensity(9)
WindOffshore.SetScaler(1)

WindOnshore = Technology(UK2, 'WindOnshore')
WindOnshore.SetCapacity(12835)
WindOnshore.SetCarbonIntensity(9)
WindOnshore.SetScaler(1)

Solar = Technology(UKSolar, 'Solar')
Solar.SetCapacity(13080)
Solar.SetCarbonIntensity(42)
Solar.SetScaler(1)

SolarDSSC = Technology(UKSolar, 'Solar')
SolarDSSC.Name = 'SolarDSSC'
SolarDSSC.SetScaler(0 * EMa)
SolarDSSC.SetCarbonIntensity(42)

SolarF = Technology(UKSolar, 'Solar')
SolarF.SetCapacity(13080)
SolarF.SetCarbonIntensity(42)
SolarF.SetScaler(1)

SolarUtility = Technology(UK2, 'SolarUtility')
SolarUtility.SetCapacity(13276)
SolarUtility.SetCarbonIntensity(42)
SolarUtility.SetScaler(1)

SolarUtilityF = Technology(UK2, 'SolarUtility')
SolarUtilityF.SetCapacity(13276)
SolarUtilityF.SetCarbonIntensity(42)
SolarUtilityF.SetScaler(1)

SolarUtilityDSSC = Technology(UK2, 'SolarUtility')
SolarUtilityDSSC.Name = 'SolarUtilityDSSC'
SolarUtilityDSSC.SetScaler(0 * EMa)
SolarUtilityDSSC.SetCarbonIntensity(42)

DC1 = DispatchClass('DC1', Nuclear, Solar, SolarDSSC)
DC2 = DispatchClass('DC2', Hydro, WindOffshore, WindOnshore, SolarUtility, SolarUtilityDSSC)
DC3 = DispatchClass('DC3', PumpedStorage)
DC4 = DispatchClass('DC4', Gas, Coal)
Total = DispatchClass('Total', Nuclear, SolarF, Hydro, SolarUtilityF, WindOffshore, WindOnshore, PumpedStorage, Gas, Coal)
Dispatched = Dispatcher(Total, DC1, DC2, DC3, DC4)
Dispatched.Dispatch()

print(np.cumsum(Dispatched.Carbon[~np.isnan(Dispatched.Carbon)])[-1])

