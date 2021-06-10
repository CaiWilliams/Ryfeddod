import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

class Original:

    def __init__(self, FilePath, Country):
        self.FilePath = FilePath
        self.Country = Country
        self.Files = os.listdir(FilePath)
        self.Days = np.zeros(0)
        self.Technologies = np.zeros(0)
        self.Data = pd.DataFrame()
        for Files in self.Files:
                File = pd.read_csv(FilePath + '//' + Files)
                self.Data = self.Data.append(File,ignore_index=True)
        if self.Country == 'NZ':
            self.TradingPeriods = np.arange(1,51,1)
            self.TradingPeriods = np.asarray([timedelta(minutes=int(30*TP)) for TP in self.TradingPeriods])
            self.Technologies = np.append(self.Technologies,pd.unique(File['Fuel_Code']))
            self.Data['Trading_date'] = [datetime.strptime(date, '%Y-%m-%d') for date in self.Data['Trading_date']]
            self.Data = self.Data.melt(['Trading_date','Site_Code','POC_Code','Nwk_Code','Gen_Code','Fuel_Code','Tech_Code'],var_name='Hour',value_name='Generation').dropna()
            self.Times = pd.Series([timedelta(minutes= int(30 * int(hour[2:]))) for hour in self.Data['Hour']])
            self.Data['Trading_date'] = self.Data['Trading_date'] + self.Times
            self.Data = self.Data.drop('Hour',axis=1).dropna()
            self.Days = self.Data['Trading_date']
            self.Data = self.Data.set_index(['Trading_date'])
        elif self.Country == 'UK':
            self.TradingPeriods = np.arange(1, 51, 1)
            self.TradingPeriods = np.asarray([timedelta(minutes=int(30 * (TP))) for TP in self.TradingPeriods])
            self.Technologies = np.append(self.Technologies, pd.unique(File.columns[4:]))
            self.Data['Settlement Date'] = [datetime.strptime(str(date), '%Y%m%d') for date in self.Data['Settlement Date']]
            self.Times = pd.Series([timedelta(minutes=int(30 * int(hour))) for hour in self.Data['SP']])
            self.Data['Settlement Date'] = self.Data['Settlement Date'] + self.Times
            self.Days = self.Data['Settlement Date']
            self.Data = self.Data.set_index(['Settlement Date'])
            self.Data = self.Data.drop(['HDR', 'SP'], axis=1)
        elif self.Country == 'UKSolar':
            self.Technologies = 'Solar'
            self.Data = self.Data.drop(['ggd_id', 'n_ggds', 'lcl_mw', 'ucl_mw', 'capacity_mwp', 'installedcapacity_mwp'], axis=1)
            self.Data['Generation'] = self.Data['generation_mw']
            self.Data = self.Data.drop('generation_mw', axis=1)
            self.Data['datetime_gmt'] = [datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%SZ') for date in self.Data['datetime_gmt']]
            self.Data = self.Data.rename(columns={'datetime_gmt': 'Settlement Date'})
            self.Times = self.Data['Settlement Date']
            self.Data = self.Data.set_index('Settlement Date')
        elif self.Country == 'UK2':
            self.Data['Settlement Period'] = [timedelta(minutes=int(Period * 30)) for Period in self.Data['Settlement Period']]
            self.Data['Settlement Date'] = pd.to_datetime(self.Data['Settlement Date'], format='%d/%m/%Y')
            self.Data['Settlement Date'] = self.Data['Settlement Date'] + self.Data['Settlement Period']
            self.Data = self.Data.drop(columns='Settlement Period')
            self.Times = self.Data['Settlement Date']
            self.Data = self.Data.set_index('Settlement Date')
        elif self.Country == 'Enhancment':
            self.Data['Date'] = [datetime.strptime(str(date), '%d/%m/%Y %H:%M') for date in self.Data['Date']]
            self.Data['Date'] = [date + timedelta(days=-1461) for date in self.Data['Date']]
            self.Data2 = self.Data.copy()
            self.Data2['Date'] = [date + timedelta(minutes=30) for date in self.Data['Date']]
            self.Data = self.Data.append(self.Data2)
            self.Data = self.Data.sort_values(by=['Date'])
            self.Data = self.Data.set_index('Date')
            #self.Data['Date'] = [for date in self.Data['Date']]

    def MatchTimes(self, MatchTo):
        self.Data = self.Data.loc[self.Data.index.isin(MatchTo.Data.index)]
        return self

class Technology:

    def __init__(self, OG, Tech):
        self.OG = OG
        self.Data = OG.Data
        self.Name = Tech
        if OG.Country == 'NZ':
            Data = self.Data[self.Data['Fuel_Code'] == Tech]
            self.Total = Data['Generation'].sum(axis=0)
            DataNS = Data.drop(['Site_Code','POC_Code','Nwk_Code','Gen_Code','Fuel_Code','Tech_Code'],axis=1).reset_index()
            self.HourlyTotal = DataNS.groupby(DataNS['Trading_date']).agg({'Generation':'sum'})
            self.DailyTotal = DataNS.groupby(DataNS['Trading_date'].dt.date).agg({'Generation':'sum'})
            self.MonthlyTotal = DataNS.groupby(DataNS['Trading_date'].dt.month).agg({'Generation':'sum'})
            self.Raw = self.Data[self.Data['Fuel_Code'] == Tech]
        elif OG.Country == 'UK':
            DataNS = self.Data.reset_index()
            DataNS.rename(columns={Tech:'Generation'},inplace=True)
            Data = self.Data[Tech]
            self.Total = Data.sum(axis=0)
            self.HourlyTotal = DataNS.groupby(DataNS['Settlement Date']).agg({'Generation':'sum'})
            self.DailyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.date).agg({'Generation': 'sum'})
            self.MonthlyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.month).agg({'Generation': 'sum'})
        elif OG.Country == 'UKSolar':
            DataNS = self.Data.reset_index()
            Data = self.Data['Generation']
            self.Total = Data.sum(axis=0)
            self.HourlyTotal = DataNS.groupby(DataNS['Settlement Date']).agg({'Generation': 'sum'})
            self.DailyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.date).agg({'Generation': 'sum'})
            self.MonthlyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.month).agg({'Generation': 'sum'})
        elif OG.Country == 'UK2':
            DataNS = self.Data.reset_index()
            Data = self.Data[Tech]
            self.Total = Data.sum(axis=0)
            DataNS.rename(columns={Tech: 'Generation'}, inplace=True)
            self.HourlyTotal = DataNS.groupby(DataNS['Settlement Date']).agg({'Generation': 'sum'})
            self.DailyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.date).agg({'Generation': 'sum'})
            self.MonthlyTotal = DataNS.groupby(DataNS['Settlement Date'].dt.month).agg({'Generation': 'sum'})

    def SetCarbonIntensity(self, CO2):
        self.CarbonIntensity = CO2
        return

    def SetCapacity(self, Cap):
        self.Capacity = Cap
        return

    def SetScaler(self, Scaler):
        self.Scaler = Scaler
        return

    def Add(self, *args):
        for a in args:
            self.HourlyTotal = self.HourlyTotal + a.HourlyTotal
        return self


class DispatchClass:

    def __init__(self, Name, *args):
        self.Name = Name
        self.Technologies = np.empty(0)
        for i in args:
            self.Technologies = np.append(self.Technologies, i)
            setattr(self, str(i.Name), i)
        #self.Technologies = args
        self.Total = 0
        self.HourlyTotal = args[0].HourlyTotal.copy()
        self.HourlyTotal['Generation'] = 0
        self.Capacity = 0
        for i in args:
            self.Total = self.Total + i.Total
            self.HourlyTotal = self.HourlyTotal + i.HourlyTotal
            #self.Capacity = self.Capacity + i.Capacity

    def CarbonIntensity(self):
        self.Carbon = self.HourlyTotal
        self.Carbon = 0
        for i in self.Technologies:
            i.Carbon = i.HourlyTotal * i.CarbonIntensity
            i.Carbon = i.Carbon.cumsum()
            self.Carbon = self.Carbon + i.Carbon
        return

class Dispatcher:

    def __init__(self, OldGen, DC1, DC2, DC3, DC4):
        self.DC1 = DC1
        self.DC2 = DC2
        self.DC3 = DC3
        self.DC4 = DC4
        self.OldGen = OldGen

    def Dispatch2(self):

        self.Dates = self.OldGen.HourlyTotal.index.to_numpy()
        self.Demand = np.ravel(self.OldGen.HourlyTotal.to_numpy())
        self.Generation = 0
        self.Carbon = 0

        for Tech in self.DC1.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            TechGeneration = TechGeneration * Tech.Scaler
            TechGeneration = np.minimum(TechGeneration, (self.Demand-self.Generation))
            self.Generation = self.Generation + TechGeneration
            self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
            setattr(self,Tech.Name, np.ravel(TechGeneration))
            setattr(self, str(Tech.Name) + "Carbon", np.ravel((TechGeneration * Tech.CarbonIntensity)))

        DC2Pre = 0#np.zeros(len(self.Demand))
        DC2Pst = 0#np.zeros(len(self.Demand))

        for Tech in self.DC2.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            DC2Pre = DC2Pre + TechGeneration
            TechGeneration = TechGeneration * Tech.Scaler
            TechGeneration = np.minimum(TechGeneration, (self.Demand-self.Generation))
            self.Generation = self.Generation + TechGeneration
            self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
            DC2Pst = DC2Pst + TechGeneration
            setattr(self, Tech.Name, np.ravel(TechGeneration))
            setattr(self, str(Tech.Name) + "Carbon", np.ravel((TechGeneration * Tech.CarbonIntensity)))

        for Tech in self.DC3.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            TechGeneration = TechGeneration * Tech.Scaler
            TechGeneration = np.minimum(TechGeneration,(self.Demand-self.Generation))
            self.StorageDischarge = TechGeneration
            self.Generation = self.Generation + TechGeneration
            setattr(self, Tech.Name, np.ravel(TechGeneration))
            setattr(self, str(Tech.Name) + "Carbon", np.ravel((TechGeneration * Tech.CarbonIntensity)))

        self.DC2Curtailed = DC2Pre - DC2Pst
        self.StorageSoC = np.zeros(len(self.Demand))
        StorageRTESQRT = 0.92
        StorageCapacity = 0#4052
        StoragePower = 500

        for i in range(0, len(self.StorageSoC)):
            self.StorageSoC[i] = np.minimum(np.abs(self.StorageSoC[i-1] + (self.DC2Curtailed[i] * StorageRTESQRT) - (self.StorageDischarge[i]/StorageRTESQRT)) , (StorageCapacity) )
        self.StorageDischarge2 = self.StorageDischarge.copy()

        for i in range(0, len(self.StorageSoC)):
            self.StorageDischarge2[i] = min((self.Demand[i] - self.Generation[i]), self.StorageSoC[i-1], StoragePower)

        self.StorageDischarge3 = self.StorageDischarge2 + self.StorageDischarge


        self.Carbon = self.Carbon + (self.StorageDischarge3 * DC3.PumpedStorage.CarbonIntensity)
        #self.Generation = self.Generation + self.StorageDischarge3


        for Tech in self.DC4.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            TechGeneration = TechGeneration * Tech.Scaler
            TechGeneration = np.minimum(TechGeneration,(self.Demand-self.Generation))
            self.Generation = self.Generation + TechGeneration
            self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
            setattr(self, Tech.Name, np.ravel(TechGeneration))
            setattr(self, str(Tech.Name) + "Carbon", np.ravel((TechGeneration * Tech.CarbonIntensity)))

        if np.any((self.Demand-self.Generation)):
            for Tech in self.DC4.Technologies:
                TechGeneration = np.minimum(Tech.Capacity,self.Demand-self.Generation)
                self.Generation = self.Generation + TechGeneration
                self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
                self.__dict__[Tech.Name] = self.__dict__[Tech.Name] + TechGeneration
                self.__dict__[str(Tech.Name)+"Carbon"] = self.__dict__[str(Tech.Name)+"Carbon"] + (TechGeneration * Tech.CarbonIntensity)


        self.Oversupply = self.Generation - self.Demand
        self.Error = np.where(self.Oversupply != 0, True, False)
        self.Carbon = (self.Carbon / self.Generation/2) * 1*10**-4
        self.Generation = self.Generation
        #self.Synchronous = ((np.ravel(self.Nuclear) + np.ravel(self.Hydro) + np.ravel(self.StorageDischarge3) + np.ravel(self.Gas) + np.ravel(self.Coal)) / np.ravel(self.Generation)) * 100
        return

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

Devices = ['Data/2015/Bangor','Data/2015/NC','Data/2015/PolySi']

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

            Dispatched.Dispatch2()

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

    #plt.plot(DSSC,NuclearU)
    #plt.plot(DSSC,SolarU)
    #plt.plot(DSSC,SolarDSSCU)
    #plt.plot(DSSC,HydroU)
    #plt.plot(DSSC,SolarUtilityU)
    #plt.plot(DSSC,SolarUtilityDSSCU)
    #plt.plot(DSSC,WindOffshoreU)
    #plt.plot(DSSC,WindOnshoreU)
    #plt.plot(DSSC,PumpedStorageU)
    #plt.plot(DSSC,CoalU)
    #plt.plot(DSSC,GasU)
    #plt.legend(labels=['Nuclear', 'Solar', 'SolarDSSC', 'Hydro', 'SolarUtility', 'SolarUtilityDSSC', 'WindOffshore','WindOnshore', 'PumpedStorage', 'Coal', 'Gas'])
    #plt.twinx()
    #plt.plot(DSSC,TotalU)

        #plt.stackplot(DSSC*100,NuclearU,SolarU,SolarDSSCU,HydroU,SolarUtilityU,SolarUtilityDSSCU,WindOffshoreU,WindOnshoreU,PumpedStorageU,CoalU,GasU,labels=['Nuclear','Solar','SolarDSSC','Hydro','SolarUtility','SolarUtilityDSSC','WindOffshore','WindOnshore','PumpedStorage','Coal','Gas'])
        #plt.plot(DSSC,TotalU)
        plt.plot(DSSC*100,(Results/Results[0])*100)
    print("Nuclear: " + str(NuclearU[-1]/TotalU[-1]))
    print("Solar: "+ str(SolarU[-1]/TotalU[-1]))
    print("Solar DSSC: " + str(SolarDSSCU[-1]/TotalU[-1]))
    print("Hydro: " + str(HydroU[-1]/TotalU[-1]))
    print("Solar Utility: " + str(SolarUtilityU[-1]/TotalU[-1]))
    print("Solar Untility DSSC: " + str(SolarUtilityDSSCU[-1]/TotalU[-1]))
    print("Wind Off Shore: " + str(WindOffshoreU[-1]/TotalU[-1]))
    print("Wind On Shore: " + str(WindOnshoreU[-1]/TotalU[-1]))
    print("Pumped Storage: " + str(PumpedStorageU[-1]/TotalU[-1]))
    print("Storage: " + str(StorageU[-1] / TotalU[-1]))
    print("Coal: " + str(CoalU[-1]/TotalU[-1]))
    print("Gas: " + str(GasU[-1]/TotalU[-1]))
    plt.xlabel('Proportion of New Technology in Grid')
    plt.ylabel('Relative Carbon Equivalent Emissions (%)')
    #plt.legend()
    plt.show()
    return

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
MeanDay(1,7)

#plt.hist(I,bins=100)
#plt.ylabel('Frequency')
#plt.xlabel('Irradiance (W/$m^2$)')
#plt.twinx()
#plt.scatter(EM.Data['Irradiance'],EM.Data['Enhancment']*EM.Data['Irradiance'],c='orange')
#plt.ylabel('Enhancment')
#plt.scatter(EM.Data['Irradiance'], SD*SolarDSSC.Scaler, label='DSSC')
#plt.scatter(EM.Data['Irradiance'], SD*Solar.Scaler, label='Installed PV')
#plt.twinx()
#plt.scatter(EM.Data['Irradiance'], SolarDSSC.Scaler*np.ones(len(EM.Data['Irradiance'])), color="tab:orange")
#plt.scatter(EM.Data['Irradiance'], S * SolarDSSC.Scaler, label='DSSC')


#print(Dispatched.Carbon)
#plt.plot(Dispatched.Generation/1000)
#print(np.cumsum(Dispatched.Carbon[~np.isnan(Dispatched.Carbon)])[-1])
#print("Nuclear: " + str(np.cumsum(Dispatched.Nuclear)[-1]/1000000/2))
#print("Solar: " + str(np.cumsum(Dispatched.Solar)[-1]/1000000/2))
#print("SolarDSSC: " + str(np.cumsum(Dispatched.SolarDSSC)[-1]/1000000/2))
#print("Hydro: " + str(np.cumsum(Dispatched.Hydro)[-1]/1000000))
#print("SolarUtility: " + str(np.cumsum(Dispatched.SolarUtility)[-1]/1000000/2))
#print("SolarUtilityDSSC: " + str(np.cumsum(Dispatched.SolarUtilityDSSC)[-1]/1000000/2))
#print("WindOffshore: " + str(np.cumsum(Dispatched.WindOffshore)[-1]/1000000/2))
#print("WindOnshore: " + str(np.cumsum(Dispatched.WindOnshore)[-1]/1000000/2))
#print("PumpedStorage: " + str(np.cumsum(Dispatched.PumpedStorage)[-1]/1000000/2))
#print("Coal: " + str(np.cumsum(Dispatched.Coal)[-1]/1000000/2))
#print("Gas: " + str(np.cumsum(Dispatched.Gas)[-1]/1000000/2))
#print("Total: " + str(np.cumsum(Dispatched.Generation)[-1]/1000000/2))
#print("Carbon: " + str(np.cumsum(Dispatched.Carbon[~np.isnan(Dispatched.Carbon)])[-1]))
#print(Dispatched.Error)
#plt.plot(Dispatched.Dates,Dispatched.Demand)
#plt.plot(Dispatched.Dates,Dispatched.Generation-Dispatched.Demand)
#plt.plot(Dispatched.Dates,Dispatched.Demand2)
#plt.twinx()
#plt.plot(Dispatched.Dates,Dispatched.Demand,color="tab:green")
#plt.plot(Dispatched.Dates,Dispatched.SolarUtilityDSSC + Dispatched.SolarDSSC,color="tab:green")
#plt.show()

EM.Data = EM.Data.replace(0,np.nan)
EM.Data = EM.Data.dropna(how='all')
W = np.arange(100,1000,100)
plt.hist(EM.Data['Irradiance'],bins=500,weights=EM.Data['Irradiance'])
plt.show()