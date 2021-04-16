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
            self.Data['Settlement Date'] = pd.to_datetime(self.Data['Settlement Date'], format='%Y-%m-%d')
            self.Data['Settlement Date'] = self.Data['Settlement Date'] + self.Data['Settlement Period']
            self.Data = self.Data.drop(columns='Settlement Period')
            self.Times = self.Data['Settlement Date']
            self.Data = self.Data.set_index('Settlement Date')

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
            self.Capacity = self.Capacity + i.Capacity

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

    def Dispatch(self):
        self.Net = -self.OldGen.HourlyTotal.to_numpy()
        self.StorageCapacity = 4630
        self.StoredEnergy = np.zeros(np.shape(self.Net))

        self.DC1.HourlyTotal = self.DC1.HourlyTotal['Generation'].to_numpy()
        self.DC2.HourlyTotal = self.DC2.HourlyTotal['Generation'].to_numpy()
        self.DC3.HourlyTotal = self.DC3.HourlyTotal['Generation'].to_numpy()
        self.DC4.HourlyTotal = self.DC4.HourlyTotal['Generation'].to_numpy()
        self.StoredEnergy = self.DC3.HourlyTotal
        idx = 0
        for Demand in self.Net:

        # DC1
            self.Net[idx] = Demand + self.DC1.HourlyTotal[idx]

            # Over Generation? Y
            if self.Net[idx] > 0:
                # Is storage full or have insufficient capacity for over generation? N
                if (self.StorageCapacity - self.StoredEnergy[idx]) >= self.Net[idx]:
                    self.StoredEnergy[idx] = self.StoredEnergy[idx] + self.Net[idx]

                # Is storage full or have insufficient capacity for over generation? Y
                else:
                    print("Inflexible Generation Needs to be Curtailed")
                    break

            # Over Generation? N
            elif self.Net[idx] == 0:
                idx = idx + 1
                continue

        # DC2
            self.Net[idx] = self.Net[idx] + self.DC2.HourlyTotal[idx]

            # Over Generation? Y
            if self.Net[idx] > 0:
                # Is storage full or have insufficient capacity for over generation? N
                if (self.StorageCapacity - self.StoredEnergy[idx]) >= self.Net[idx]:
                    self.StoredEnergy[idx] = self.StoredEnergy[idx] + self.Net[idx]
                    print("Curtail Flexible Low Carbon Generation")
                # Is storage full or have insufficient capacity for over generation? Y
                else:
                    print("Curtail Flexible Low Carbon Generation")
                    continue

            # Over Generation? N
            elif self.Net[idx] <= 0:
                if self.Net[idx] == 0:
                    idx = idx + 1
                    continue

        # DC3

            # Energy storage empty? N
            if self.StoredEnergy[idx] == 0:

                self.Net[idx] = self.Net[idx] + self.DC4.HourlyTotal[idx]

                if self.Net[idx] == 0:
                    idx = idx + 1
                    continue
                else:
                    print("Insufficient Plant to meet demand")
                idx = idx + 1
                continue
            else:
                # Discharge storage to minimum of: - unmet demand -storage rating -remaining power that can be delivered
                if self.StoredEnergy[idx] >= np.abs(self.Net[idx]):
                    self.StoredEnergy[idx] = self.StoredEnergy[idx] + self.Net[idx]
                    self.Net[idx] = 0
                elif self.StoredEnergy[idx] <= np.abs(self.Net[idx]):
                    self.Net[idx] = self.Net[idx] + self.StoredEnergy[idx]
                    self.StoredEnergy[idx] = 0


            if self.Net[idx] == 0:
                idx = idx + 1
                continue

        #DC4
            self.Net[idx] = self.Net[idx] + self.DC4.HourlyTotal[idx]
            idx = idx + 1

        return

UK = Original('Data/UK','UK')
UKSolar = Original('Data/UKSolar','UKSolar')
UK2 = Original('Data/UK2', 'UK2')

UK = UK.MatchTimes(UK2)
UKSolar = UKSolar.MatchTimes(UK2)
UK = UK.MatchTimes(UKSolar)
UK2 = UK2.MatchTimes(UKSolar)

Biomass = Technology(UK2, 'Biomass')
Biomass.SetCapacity(4237)
Biomass.SetCarbonIntensity(230)

Gas = Technology(UK2, 'Gas')
Gas.SetCapacity(38274)
Gas.SetCarbonIntensity(443)

Coal = Technology(UK2, 'Coal')
Coal.SetCapacity(6780)
Coal.SetCarbonIntensity(960)

PumpedStorage = Technology(UK2, 'PumpedStorage')
PumpedStorage.SetCapacity(4052)
PumpedStorage.SetCarbonIntensity(12)

Hydro = Technology(UK2, 'Hydro')
Hydro.SetCapacity(1882)
Hydro.SetCarbonIntensity(10)

Nuclear = Technology(UK2, 'Nuclear')
Nuclear.SetCapacity(8209)
Nuclear.SetCarbonIntensity(12)

SolarUtility = Technology(UK2, 'SolarUtility')
SolarUtility.SetCapacity(13276)
SolarUtility.SetCarbonIntensity(42)

WindOffshore = Technology(UK2, 'WindOffshore')
WindOffshore.SetCapacity(10365)
WindOffshore.SetCarbonIntensity(9)

WindOnshore = Technology(UK2, 'WindOnshore')
WindOnshore.SetCapacity(12835)
WindOnshore.SetCarbonIntensity(9)

Solar = Technology(UKSolar, 'Solar')
Solar.SetCapacity(13080)
Solar.SetCarbonIntensity(42)

Interconect = Technology(UK, 'INTFR')

INTIRL = Technology(UK, 'INTIRL')
INTNED = Technology(UK, 'INTNED')
INTEW = Technology(UK, 'INTEW')
INTNEM = Technology(UK, 'INTNEM')
INTELEC = Technology(UK, 'INTELEC')
INTIFA2 = Technology(UK, 'INTIFA2')
INTNSL = Technology(UK, 'INTNSL')

Interconect = Interconect.Add(INTIRL,INTNED,INTEW,INTNEM,INTELEC,INTIFA2,INTNSL)
Interconect.SetCarbonIntensity(42)
Interconect.SetCapacity(8209)




DC1 = DispatchClass('DC1', Nuclear, Solar, Biomass)
DC2 = DispatchClass('DC2', Hydro, SolarUtility, WindOffshore, WindOnshore)
DC3 = DispatchClass('DC3', PumpedStorage)
DC4 = DispatchClass('DC4', Coal, Gas, Interconect)

Total = DispatchClass('Total', Nuclear, Solar, Biomass, Hydro, SolarUtility, WindOffshore, WindOnshore, PumpedStorage, Coal, Gas, Interconect)
#Total = DispatchClass('Total', Nuclear, Solar, Biomass)

Dispatched = Dispatcher(Total, DC1, DC2, DC3, DC4)
Dispatched.Dispatch()

plt.plot(Dispatched.Net )
plt.show()


#plt.stackplot(Total.HourlyTotal.index, Total.Biomass.HourlyTotal['Generation'], Total.Nuclear.HourlyTotal['Generation'], Total.WindOffshore.HourlyTotal['Generation'], Total.WindOnshore.HourlyTotal['Generation'], Total.SolarUtility.HourlyTotal['Generation'], Total.Solar.HourlyTotal['Generation'], Total.Hydro.HourlyTotal['Generation'], Total.PumpedStorage.HourlyTotal['Generation'], Total.Gas.HourlyTotal['Generation'], Total.Coal.HourlyTotal['Generation'], Total.INTFR.HourlyTotal['Generation'], labels=['Biomass','Nuclear','Wind OffShore','Wind OnShore','Solar Utility', 'Solar Home', 'Hydo', 'Pumped Storage', 'Gas', 'Coal', 'Interconect'])
#plt.stackplot(DC1.HourlyTotal.index,DC1.HourlyTotal['Generation'],DC2.HourlyTotal['Generation'],DC3.HourlyTotal['Generation'],DC4.HourlyTotal['Generation'],labels=['DC1','DC2','DC3','DC4'])
#plt.legend()
#plt.show()
