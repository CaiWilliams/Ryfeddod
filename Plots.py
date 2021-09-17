import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from Main import *
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

Devices = ['Data/2015/Bangor']

def Sweep(Min, Max, Steps,Devices):

    DSSC = np.linspace(Min,Max,Steps)
    #DSSC =np.zeros(Steps)
    NDSSC = np.linspace(1,0,Steps)
    Results = np.zeros(len(DSSC))
    NuclearU = np.zeros(len(DSSC))
    SolarU = np.zeros(len(DSSC))
    SolarDSSCU = np.zeros(len(DSSC))
    HydroU = np.zeros(len(DSSC))
    SolarUtilityU = np.zeros(len(DSSC))
    SolarUtilityDSSCU = np.zeros(len(DSSC))
    WindOffshoreU = np.zeros(len(DSSC))
    WindOnshoreU = np.zeros(len(DSSC))
    PumpedStorageU = np.zeros(len(DSSC))
    StorageU = np.zeros(len(DSSC))
    CoalU = np.zeros(len(DSSC))
    GasU = np.zeros(len(DSSC))
    TotalU = np.zeros(len(DSSC))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    for Device in Devices:
        UKSolar = Original('Data/2015/UKSolar', 'UKSolar')
        UK2 = Original('Data/2015/UK2', 'UK2')
        EM = Original(Device, 'Enhancment')
        UKSolar = UKSolar.MatchTimes(UK2)
        UK2 = UK2.MatchTimes(UKSolar)
        EM = EM.MatchTimes(UK2)
        EM = EM.MatchTimes(UKSolar)
        UKSolar = UKSolar.MatchTimes(EM)
        UK2 = UK2.MatchTimes(EM)
        EMa = np.ravel(EM.Data['Enhancment'].to_numpy())
        for idx,Scale in enumerate(DSSC):

            Solar.SetScaler(NDSSC[idx])
            SolarUtility.SetScaler(NDSSC[idx])

            SolarDSSC.SetScaler(Scale * EMa)
            SolarUtilityDSSC.SetScaler(Scale * EMa)

            Dispatched.Dispatch()

            Results[idx] = np.cumsum(Dispatched.Carbon[~np.isnan(Dispatched.Carbon)])[-1]

            NuclearU[idx] = np.cumsum(Dispatched.Nuclear)[-1]/1000000/2
            SolarU[idx] = np.cumsum(Dispatched.Solar)[-1]/1000000/2
            SolarDSSCU[idx] = np.cumsum(Dispatched.SolarDSSC)[-1]/1000000/2
            HydroU[idx] = np.cumsum(Dispatched.Hydro)[-1]/1000000/2
            SolarUtilityU[idx] = np.cumsum(Dispatched.SolarUtility)[-1]/1000000/2
            SolarUtilityDSSCU[idx] = np.cumsum(Dispatched.SolarUtilityDSSC)[-1]/1000000/2
            WindOffshoreU[idx] = np.cumsum(Dispatched.WindOffshore)[-1]/1000000/2
            WindOnshoreU[idx] = np.cumsum(Dispatched.WindOnshore)[-1]/1000000/2
            PumpedStorageU[idx] = np.cumsum(Dispatched.PumpedStorage)[-1]/1000000/2
            StorageU[idx] = np.cumsum(Dispatched.StorageDischarge3)[-1]/1000000/2
            CoalU[idx] = np.cumsum(Dispatched.Coal)[-1]/1000000/2
            GasU[idx] = np.cumsum(Dispatched.Gas)[-1]/1000000/2
            TotalU[idx] = np.cumsum(Dispatched.Generation)[-1]/1000000/2

    plt.stackplot(DSSC,NuclearU, SolarU, SolarDSSCU, HydroU, SolarUtilityU,SolarUtilityDSSCU,WindOffshoreU,WindOnshoreU,PumpedStorageU,StorageU,CoalU,GasU)
    plt.show()

def MeanMonths(Scale):
    UKSolar = Original('Data/2015/UKSolar', 'UKSolar')
    UK2 = Original('Data/2015/UK2', 'UK2')
    EM = Original('Data/2015/Bangor', 'Enhancment')
    UKSolar = UKSolar.MatchTimes(UK2)
    UK2 = UK2.MatchTimes(UKSolar)
    EM = EM.MatchTimes(UK2)
    EM = EM.MatchTimes(UKSolar)
    UKSolar = UKSolar.MatchTimes(EM)
    UK2 = UK2.MatchTimes(EM)
    EMa = np.ravel(EM.Data['Enhancment'].to_numpy())


    Solar.SetScaler(1-Scale)
    SolarUtility.SetScaler(1-Scale)

    SolarDSSC.SetScaler(Scale * EMa)
    SolarUtilityDSSC.SetScaler(Scale * EMa)

    Dispatched.Dispatch2()
    Dt = pd.DataFrame()
    Dt['Dates'] = Dispatched.Dates
    Dt['Nuclear'] = Dispatched.Nuclear
    Dt['Solar'] = Dispatched.Solar
    Dt['SolarDSSC'] = Dispatched.SolarDSSC
    Dt['Hydro'] = Dispatched.Hydro
    Dt['SolarUtility'] = Dispatched.SolarUtility
    Dt['SolarUtilityDSSC'] = Dispatched.SolarUtilityDSSC
    Dt['WindOffShore'] = Dispatched.WindOffshore
    Dt['WindOnShore'] = Dispatched.WindOnshore
    Dt['PumpedStorage'] = Dispatched.PumpedStorage
    Dt['Coal'] = Dispatched.Coal
    Dt['Gas'] = Dispatched.Gas

    NuclearG = np.zeros(12)
    SolarG = np.zeros(12)
    SolarDSSCG = np.zeros(12)
    HydroG = np.zeros(12)
    SolarUtilityG = np.zeros(12)
    SolarUtilityDSSCG = np.zeros(12)
    WindOffshoreG = np.zeros(12)
    WindOnshoreG = np.zeros(12)
    PumpedStorageG = np.zeros(12)
    CoalG = np.zeros(12)
    GasG = np.zeros(12)
    Months = np.arange(1,13,1)
    for idx,Month in enumerate(Months):
        NuclearG[idx] = Dt[Dt['Dates'].dt.month == Month]['Nuclear'].mean()
        SolarG[idx] = Dt[Dt['Dates'].dt.month == Month]['Solar'].mean()
        SolarDSSCG[idx] = Dt[Dt['Dates'].dt.month == Month]['SolarDSSC'].mean()
        HydroG[idx] = Dt[Dt['Dates'].dt.month == Month]['Hydro'].mean()
        SolarUtilityG[idx] = Dt[Dt['Dates'].dt.month == Month]['SolarUtility'].mean()
        SolarUtilityDSSCG[idx] = Dt[Dt['Dates'].dt.month == Month]['SolarUtilityDSSC'].mean()
        WindOffshoreG[idx] = Dt[Dt['Dates'].dt.month == Month]['WindOffShore'].mean()
        WindOnshoreG[idx] = Dt[Dt['Dates'].dt.month == Month]['WindOnShore'].mean()
        PumpedStorageG[idx] = Dt[Dt['Dates'].dt.month == Month]['PumpedStorage'].mean()
        CoalG[idx] = Dt[Dt['Dates'].dt.month == Month]['Coal'].mean()
        GasG[idx] = Dt[Dt['Dates'].dt.month == Month]['Gas'].mean()

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    plt.stackplot(Months, NuclearG,SolarG,SolarDSSCG,HydroG,SolarUtilityG,SolarUtilityDSSCG,WindOffshoreG,WindOnshoreG,PumpedStorageG,CoalG,GasG)
    plt.show()
    return

def MeanDay(Scale,Month):
    UKSolar = Original('Data/2015/UKSolar', 'UKSolar')
    UK2 = Original('Data/2015/UK2', 'UK2')
    EM = Original('Data/2015/Bangor', 'Enhancment')
    UKSolar = UKSolar.MatchTimes(UK2)
    UK2 = UK2.MatchTimes(UKSolar)
    EM = EM.MatchTimes(UK2)
    EM = EM.MatchTimes(UKSolar)
    UKSolar = UKSolar.MatchTimes(EM)
    UK2 = UK2.MatchTimes(EM)
    EMa = np.ravel(EM.Data['Enhancment'].to_numpy())

    Solar.SetScaler(1 - Scale)
    SolarUtility.SetScaler(1 - Scale)

    SolarDSSC.SetScaler(Scale * EMa)
    SolarUtilityDSSC.SetScaler(Scale * EMa)

    Dispatched.Dispatch2()
    Dt = pd.DataFrame()
    Dt['Dates'] = Dispatched.Dates
    Dt['Nuclear'] = Dispatched.NuclearCarbon
    Dt['Solar'] = Dispatched.SolarCarbon
    Dt['SolarDSSC'] = Dispatched.SolarDSSCCarbon
    Dt['Hydro'] = Dispatched.HydroCarbon
    Dt['SolarUtility'] = Dispatched.SolarUtilityCarbon
    Dt['SolarUtilityDSSC'] = Dispatched.SolarUtilityDSSCCarbon
    Dt['WindOffShore'] = Dispatched.WindOffshoreCarbon
    Dt['WindOnShore'] = Dispatched.WindOnshoreCarbon
    Dt['PumpedStorage'] = Dispatched.PumpedStorageCarbon
    Dt['Coal'] = Dispatched.CoalCarbon
    Dt['Gas'] = Dispatched.GasCarbon

    NuclearG = np.zeros(24)
    SolarG = np.zeros(24)
    SolarDSSCG = np.zeros(24)
    HydroG = np.zeros(24)
    SolarUtilityG = np.zeros(24)
    SolarUtilityDSSCG = np.zeros(24)
    WindOffshoreG = np.zeros(24)
    WindOnshoreG = np.zeros(24)
    PumpedStorageG = np.zeros(24)
    CoalG = np.zeros(24)
    GasG = np.zeros(24)
    Hours = np.arange(0, 24, 1)

    Dt = Dt[Dt['Dates'].dt.month == Month]
    for idx,Hour in enumerate(Hours):
        NuclearG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['Nuclear'].mean()
        SolarG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['Solar'].mean()
        SolarDSSCG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['SolarDSSC'].mean()
        HydroG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['Hydro'].mean()
        SolarUtilityG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['SolarUtility'].mean()
        SolarUtilityDSSCG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['SolarUtilityDSSC'].mean()
        WindOffshoreG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['WindOffShore'].mean()
        WindOnshoreG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['WindOnShore'].mean()
        PumpedStorageG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['PumpedStorage'].mean()
        CoalG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['Coal'].mean()
        GasG[idx] = Dt[Dt['Dates'].dt.hour == Hour]['Gas'].mean()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    plt.stackplot(Hours, NuclearG, SolarG, SolarDSSCG, HydroG, SolarUtilityG, SolarUtilityDSSCG, WindOffshoreG,WindOnshoreG, PumpedStorageG, CoalG, GasG)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Mean Power Generation (MW)")
    plt.show()
    return

#Sweep(0,1,10,Devices)
#MeanMonths(0.5)
#MeanDay(1,7)