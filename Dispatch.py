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
            Times = pd.Series([timedelta(minutes= int(30 * int(hour[2:]))) for hour in self.Data['Hour']])
            #print(np.unique(Times))
            self.Data['Trading_date'] = self.Data['Trading_date'] + Times
            self.Data = self.Data.drop('Hour',axis=1).dropna()
            self.Days = self.Data['Trading_date']
            self.Data = self.Data.set_index(['Trading_date'])
        elif self.Country == 'UK':
            self.TradingPeriods = np.arange(1, 51, 1)
            self.TradingPeriods = np.asarray([timedelta(minutes=int(30 * TP)) for TP in self.TradingPeriods])
            self.Technologies = np.append(self.Technologies, pd.unique(File.columns[4:]))
            self.Data['Settlement Date'] = [datetime.strptime(str(date), '%Y%m%d') for date in self.Data['Settlement Date']]
            Times = pd.Series([timedelta(minutes=int(30 * int(hour))) for hour in self.Data['SP']])
            self.Data['Settlement Date'] = self.Data['Settlement Date'] + Times
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
            self.Data = self.Data.set_index('Settlement Date')

class Technology(Original):

    def __init__(self, OG, Tech):
        super().__init__(OG.FilePath,OG.Country)
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

    
    def __rmul__(self, other):
        return self.Total * other

    def SetCarbonIntensity(self, CO2):
        self.CarbonIntensity = CO2
        return
    

class DispatchClass:

    def __init__(self, *args):
        self.Technologies = np.empty(0)
        for i in args:
            self.Technologies = np.append(self.Technologies, i)
        #self.Technologies = args
        self.Total = 0
        self.HourlyTotal = args[0].HourlyTotal.copy()
        self.HourlyTotal['Generation'] = 0
        for i in self.Technologies:
            self.Total = self.Total + i.Total
            self.HourlyTotal = self.HourlyTotal.add(i.HourlyTotal)

class Dispatcher:

    def __init__(self, OldGen, *args):
        self.RankedClasses = args
        self.OldGen = OldGen

    def Dispatch(self):
        self.NewGen = self.OldGen.HourlyTotal.copy()
        self.NewGen['Generation'] = 0
        Timestep = self.OldGen.HourlyTotal.reset_index()['Settlement Date'].to_numpy()
        OldGen = self.OldGen.HourlyTotal['Generation'].to_numpy()
        for Demand,T in zip(OldGen, Timestep):
            Generation = 0
            for Class in self.RankedClasses:
                Generation = Generation + Class.HourlyTotal[Class.HourlyTotal.index == T]['Generation'].to_numpy()[0]
                UnmetDemand = Generation - Demand
                if UnmetDemand > 0:
                    if StoredGeneration <= UnmetDemand:
                        Generation = Generation + UnmetDemand
                        StoredGeneration = StoredGeneration - UnmetDemand
                elif UnmetDemand < 0:
                    if StoredGeneration <= StorageCapacity:
                        if UnmetDemand > StoredCapacity:
                            StoredGeneration = StoredGeneration - StoredCapacity
                            Generation = Generation + StoredCapacity
                        else:
                            StoredGeneration = StoredGeneration - UnmetDemand
                            Generation = Generation + UnmetDemand
            print(UnmetDemand)
        return

UK = Original('Data/UK', 'UK')
UKSolar = Original('Data/UKSolar','UKSolar')

NPSHyd = Technology(UK, 'NPSHyd')
NPSHyd.SetCarbonIntensity(24)

PS = Technology(UK, 'PS')
CCGT = Technology(UK, 'CCGT')
OCGT = Technology(UK, 'OCGT')
Coal = Technology(UK, 'Coal')
Coal.SetCarbonIntensity(820)
Oil = Technology(UK, 'Oil')
Nuclear = Technology(UK, 'Nuclear')
Wind = Technology(UK, 'Wind')
Other = Technology(UK, 'Other')
Biomass = Technology(UK,'Biomass')
INTFR = Technology(UK,'INTFR')
INTIRL = Technology(UK,'INTIRL')
INTNED = Technology(UK,'INTNED')
INTEW = Technology(UK, 'INTEW')
INTNEM = Technology(UK, 'INTNEM')
INTELEC = Technology(UK, 'INTELEC')
INTIFA2 = Technology(UK, 'INTIFA2')
INTNSL = Technology(UK, 'INTNSL')
Solar = Technology(UKSolar, 'Solar')
print(Solar.Total)




DC1 = DispatchClass(Nuclear, Biomass, Other)
DC2 = DispatchClass(NPSHyd, Wind, Solar)
DC3 = DispatchClass(PS)
DC4 = DispatchClass(INTFR, INTIRL, INTNED, INTEW, INTNEM, INTELEC, INTIFA2, INTNSL, CCGT, OCGT)

#OldGen = DispatchClass(NPSHyd,PS,CCGT,OCGT,Oil,Coal,Nuclear,Wind,Other,INTFR,INTIRL,INTNED,INTEW,INTNEM,INTELEC,INTIFA2,INTNSL)
#Dispatched = Dispatcher(DC1,DC2,DC3,DC4)
#Dispatched.Dispatch()

plt.stackplot(DC1.HourlyTotal.index,DC1.HourlyTotal['Generation'],DC2.HourlyTotal['Generation'],DC3.HourlyTotal['Generation'],DC4.HourlyTotal['Generation'],labels=['DC1','DC2','DC3','DC4'])
#plt.legend()
plt.show()
