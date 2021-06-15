import difflib

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
import pytz
import requests
import io
from pvlive_api import PVLive
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy

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

class Grid:

    def __init__(self,MixDir):
        self.BMRSKey = "zz6sqbg3mg0ybyc"
        with open(MixDir) as Mix_File:
            self.Mix = json.load(Mix_File)

        DataSources = set()
        for Tech in self.Mix["Technologies"]:
            DataSources.add(Tech["Source"])

        self.StartDate = datetime.strptime(self.Mix['StartDate'],'%Y-%m-%d')
        self.EndDate = datetime.strptime(self.Mix['EndDate'], '%Y-%m-%d')

        if self.Mix["Country"] == "UK":
            if "BMRS" in DataSources:
                self.BMRSFetch()
            if "PVLive" in DataSources:
                self.PVLiveFetch()

            for Tech in self.Mix['Technologies']:
                if Tech['Source'] == "BMRS":
                    Tech['Generation'] = self.BMRSData[str(Tech['Technology'])]
                    Tech['Generation'] = Tech['Generation'].rename('Generation')
                if Tech['Source'] == "PVLive":
                    Tech['Generation'] = self.PVLiveData['generation_mw']
                    Tech['Generation'] = Tech['Generation'].rename('Generation')

    def BMRSFetch(self):
        NumDays = (self.EndDate - self.StartDate).days
        Days = [self.StartDate + timedelta(days=1 * Day) for Day in range(0,NumDays+1)]
        DaysStr = [Day.strftime('%Y-%m-%d') for Day in Days]
        AllAPIRequests = ['https://api.bmreports.com/BMRS/B1620/V1?APIKey='+ self.BMRSKey +'&SettlementDate=' + SettlementDate + '&Period=*&ServiceType=csv'for SettlementDate in DaysStr]
        AllAPIAnswers = [requests.get(APIrequest) for APIrequest in AllAPIRequests]
        ALLAPIDataframes = [pd.read_csv(io.StringIO(Answer.text), skiprows=[0, 1, 2, 3], skipfooter=1, engine='python',index_col=False).sort_values('Settlement Period') for Answer in AllAPIAnswers]
        YearDataframe = pd.concat(ALLAPIDataframes, ignore_index=True)
        YearDataframe = YearDataframe.drop(columns=['*Document Type', 'Business Type', 'Process Type', 'Time Series ID', 'Curve Type', 'Resolution','Active Flag', 'Document ID', 'Document RevNum'])
        YearDataframe = YearDataframe.pivot_table(index=['Settlement Date', 'Settlement Period'], columns='Power System Resource  Type', values='Quantity')
        YearDataframe = YearDataframe.reset_index()
        YearDataframe["Settlement Period"] = [timedelta(minutes=int(Period * 30)) for Period in YearDataframe['Settlement Period']]
        YearDataframe['Settlement Date'] = pd.to_datetime(YearDataframe['Settlement Date'], format='%Y-%m-%d')
        YearDataframe['Settlement Date'] = YearDataframe['Settlement Date'] + YearDataframe['Settlement Period']
        YearDataframe = YearDataframe.drop(columns=['Settlement Period'])
        self.Dates = YearDataframe['Settlement Date']
        YearDataframe = YearDataframe.set_index("Settlement Date")
        YearDataframe = YearDataframe.fillna(0)
        self.BMRSData = YearDataframe
        return self.BMRSData

    def PVLiveFetch(self):
        pvl = PVLive()
        tz = pytz.timezone('Europe/London')
        self.StartDate = tz.localize(self.StartDate)
        self.EndDate = tz.localize(self.EndDate)
        self.PVLiveData = pvl.between(self.StartDate,self.EndDate,dataframe=True)
        self.PVLiveData['datetime_gmt'] = [t.replace(tzinfo=None) for t in self.PVLiveData['datetime_gmt']]
        self.PVLiveData = self.PVLiveData.sort_values(by=['datetime_gmt'])
        self.PVLiveData = self.PVLiveData.set_index('datetime_gmt')
        self.PVLiveData = self.PVLiveData.fillna(0)
        self.PVLiveData.index = self.PVLiveData.index.rename('Settlement Date')
        self.PVLiveData.to_csv("BTM.csv")
        return self.PVLiveData

    def Add(self, Name, Tech):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset_Copy = Asset.copy()
                Asset_Copy['Technology'] = Name
                self.Mix['Technologies'].append(Asset_Copy)
                return self

    def Modify(self,Tech,**kwags):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset.update(kwags)
        return self

    def PVGISFetch(self,EnhancmentDir,Latitude,Longitude):
        Startyear = self.StartDate.year
        EndYear = self.EndDate.year
        PVGISAPICall = "https://re.jrc.ec.europa.eu/api/seriescalc?lat=" + str(Latitude) + "&lon=" + str(Longitude) + "&startyear=" + str(Startyear) + "&endyear=" + str(EndYear) + "&outputformat=csv"
        PVGISAnswer = requests.get(PVGISAPICall)
        PVGISData = pd.read_csv(io.StringIO(PVGISAnswer.text), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7],engine='python', usecols=['time', 'G(i)'])
        PVGISData['time'] = pd.to_datetime(PVGISData['time'], format='%Y%m%d:%H%M')
        PVGISData['time'] = [t.replace(minute=0) for t in PVGISData['time']]

        HalfHours = PVGISData.copy()
        HalfHours['time'] = HalfHours['time'] + timedelta(minutes=30)
        #MeanIrradiance = np.zeros(len(HalfHours['time']))


        #for idx, t in enumerate(HalfHours['time']):

        Before = HalfHours['time'][:] - timedelta(minutes=30)
        After = HalfHours['time'][:] + timedelta(minutes=30)

        IrradianceAfter = PVGISData.loc[PVGISData['time'].isin(After[:])]['G(i)'].to_numpy()
        IrradianceAfter = np.append(IrradianceAfter,0)
        IrradianceBefore = PVGISData.loc[PVGISData['time'].isin(Before[:])]['G(i)'].to_numpy()


        MeanIrradiance = (IrradianceBefore[:] + IrradianceAfter[:])/2

        HalfHours['G(i)'] = MeanIrradiance

        PVGISData = pd.concat([PVGISData, HalfHours])
        PVGISData = PVGISData.sort_values(by=['time'])
        PVGISData = PVGISData.set_index(['time'])
        PVGISData.index = PVGISData.index.rename('Settlement Date')

        IndexValues = [Asset['Generation'].index.values for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))
        PVGISData = PVGISData.loc[PVGISData.index.isin(CommonIndex)]

        Enhancment = pd.read_csv(EnhancmentDir)
        f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
        DynamScale = f(PVGISData['G(i)'])

        return DynamScale

    def DynamicScaleingPVGIS(self, Tech, DynamScale, BaseScale):

        Scale = DynamScale * BaseScale

        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset['Scaler'] = Scale[:]

        return self

    def MatchDates(self):

        IndexValues = [Asset['Generation'].index.values for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set,IndexValues)))

        Lengths = np.zeros(len(self.Mix['Technologies']))
        for idx,Asset in enumerate(self.Mix['Technologies']):
            Asset['Generation'] = Asset['Generation'].loc[Asset['Generation'].index.isin(CommonIndex)]
            Asset['Generation'] = Asset['Generation'][~Asset['Generation'].index.duplicated(keep='first')]
            self.Dates = Asset['Generation'].index
            Lengths[idx] = len(Asset['Generation'].index)

        return self

    def Demand(self):
        self.Demand = pd.DataFrame(index = self.Mix['Technologies'][0]['Generation'].index.copy())
        self.Demand = 0
        for Asset in self.Mix['Technologies']:
            self.Demand = self.Demand + Asset['Generation'][:]
        return self

    def CarbonEmissions(self):
        self.CarbonEmissions = pd.DataFrame(index = self.Mix['Technologies'][0]['Generation'].index.copy())
        self.CarbonEmissions = 0
        #self.CarbonEmissions['Generation'] = self.CarbonEmissions['Generation'].rename('CO2E')

        for Asset in self.Mix['Technologies']:
            self.CarbonEmissions = self.CarbonEmissions + (Asset['Generation'][:] * Asset['CarbonIntensity'])
        return self

    def Save(self,dir,Filename):
        with open(str(dir)+'\\'+str(Filename)+'.NGM','wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def Load(dir):
        with open(dir,'rb') as handle:
            return pickle.load(handle)

class Dispatch:

    def __init__(self,NG):
        self.NG = NG
        self.Distributed = copy.copy(NG)
        self.Demand = self.NG.Demand
        self.Generation = self.NG.Demand
        self.Generation = 0
        self.CarbonEmissions = self.NG.Demand
        self.CarbonEmissions = 0
        self.Order()
        self.Distribute(self.DC1)
        self.Distribute(self.DC2)
        self.Distribute(self.DC3)

        self.Storage()
        self.Distribute(self.DC4)
        self.Undersuply()
        self.Misc()

    def Order(self):

        self.DC1 = np.zeros(0)
        self.DC2 = np.zeros(0)
        self.DC3 = np.zeros(0)
        self.DC4 = np.zeros(0)

        for Asset in self.NG.Mix['Technologies']:
            if Asset['DispatchClass'] == 4:
                self.DC4 = np.append(self.DC4, Asset)
            elif Asset['DispatchClass'] == 3:
                self.DC3 = np.append(self.DC3, Asset)
            elif Asset['DispatchClass'] == 2:
                self.DC2 = np.append(self.DC2, Asset)
            elif Asset['DispatchClass'] == 1:
                self.DC1 = np.append(self.DC1, Asset)
        return

    def Distribute(self,DC):
        for Asset in DC:
            DemandRemaining = 0
            MaxGen = Asset['Generation'] * Asset['Scaler']
            DemandRemaining = self.Demand - self.Generation
            Gen = np.minimum(MaxGen, DemandRemaining)
            self.Generation = self.Generation + Gen
            self.CarbonEmissions = self.CarbonEmissions + (Gen * Asset['CarbonIntensity'])
            for DissributedAsset in self.Distributed.Mix['Technologies']:
                if Asset['Technology'] == DissributedAsset['Technology']:
                    DissributedAsset['Generation'] = Gen
                    DissributedAsset['CarbonEmissions'] = Gen * Asset['CarbonIntensity']
        return

    def Undersuply(self):
        if np.any((self.Demand-self.Generation)):
            for Asset in self.DC4:
                MaxGen = Asset['Capacity']
                DemandRemaning = self.Demand - self.Generation
                Gen = np.minimum(MaxGen, DemandRemaning)
                self.Generation = self.Generation + Gen
                self.CarbonEmissions = self.CarbonEmissions + (Gen * Asset['CarbonIntensity'])
                for DissributedAsset in self.Distributed.Mix['Technologies']:
                    if Asset['Technology'] == DissributedAsset['Technology']:
                        DissributedAsset['Generation'] = DissributedAsset['Generation'] + Gen
                        DissributedAsset['CarbonEmissions'] = DissributedAsset['CarbonEmissions'] +  (Gen * Asset['CarbonIntensity'])
        return

    def Storage(self):

        StorageRTESQRT = 0.92
        StoragePower = 500

        for Asset in self.NG.Mix['Technologies']:
            if Asset['Technology'] == "Hydro Pumped Storage":
                StorageCapacity = Asset['Capacity']

        Pre = 0
        Post = 0

        for Asset in self.DC2:

            for AssetPre in self.NG.Mix['Technologies']:
                if Asset['Technology'] == AssetPre['Technology']:
                    Pre = Pre + np.ravel(AssetPre['Generation'].to_numpy(na_value=0))

            for AssetPost in self.Distributed.Mix['Technologies']:
                if Asset['Technology'] == AssetPost['Technology']:
                    Post = Post + np.ravel(AssetPost['Generation'].to_numpy(na_value=0))

        DC2Curtailed = Pre - Post

        if np.sum(DC2Curtailed) == 0:
            return self
        else:
            print("A")

        return self

    def Misc(self):
        self.Oversuply = self.Generation - self.Demand
        self.Error = np.where(self.Oversuply != 0, True, False)

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

    def Distribute(self,DC):
        for Tech in DC.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            TechGeneration = TechGeneration * Tech.Scaler
            TechGeneration = np.minimum(TechGeneration, (self.Demand-self.Generation))
            self.Generation = self.Generation + TechGeneration
            self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
            setattr(self, Tech.Name, np.ravel(TechGeneration))
            setattr(self, Tech.Name + "Carbon", np.ravel((TechGeneration * Tech.CarbonIntensity)))
        return

    def Dispatch(self):

        self.Dates = self.OldGen.HourlyTotal.index.to_numpy()
        self.Demand = np.ravel(self.OldGen.HourlyTotal.to_numpy())
        self.Generation = 0
        self.Carbon = 0

        self.Distribute(self.DC1)

        DC2Pre = 0
        for Tech in self.DC2.Technologies:
            TechGeneration = np.ravel(Tech.HourlyTotal.to_numpy())
            DC2Pre = DC2Pre + TechGeneration

        self.Distribute(self.DC2)

        DC2Pst = 0
        for Tech in self.DC2.Technologies:
            TechGeneration = self.__dict__[Tech.Name]
            DC2Pst = DC2Pst + TechGeneration

        self.Distribute(self.DC3)

        self.StorageDischarge = 0
        for Tech in self.DC3.Technologies:
            TechGeneration = self.__dict__[Tech.Name]
            self.StorageDischarge = self.StorageDischarge + TechGeneration

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

        self.Carbon = self.Carbon + (self.StorageDischarge3 * self.DC3.PumpedStorage.CarbonIntensity)
        #self.Generation = self.Generation + self.StorageDischarge3

        self.Distribute(self.DC4)

        if np.any((self.Demand-self.Generation)):
            for Tech in self.DC4.Technologies:
                TechGeneration = np.minimum(Tech.Capacity,self.Demand-self.Generation)
                self.Generation = self.Generation + TechGeneration
                self.Carbon = self.Carbon + (TechGeneration * Tech.CarbonIntensity)
                self.__dict__[Tech.Name] = self.__dict__[Tech.Name] + TechGeneration
                self.__dict__[str(Tech.Name)+"Carbon"] = self.__dict__[str(Tech.Name)+"Carbon"] + (TechGeneration * Tech.CarbonIntensity)


        self.Oversupply = self.Generation - self.Demand
        self.Error = np.where(self.Oversupply != 0, True, False)
        #self.Carbon = (self.Carbon) * 1*10**-4
        self.Generation = self.Generation
        #self.Synchronous = ((np.ravel(self.Nuclear) + np.ravel(self.Hydro) + np.ravel(self.StorageDischarge3) + np.ravel(self.Gas) + np.ravel(self.Coal)) / np.ravel(self.Generation)) * 100
        return


def SweepSolarGen(NG, Start, Stop, Steps, Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    DynamScale = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)

    GenSum = np.ndarray(shape = (len(NG.Mix['Technologies']),len(Existing)))

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Grid.Load('Data/2016.NGM')
        NG = NG.Modify('Solar', Scaler=Existing[idx])
        NG = NG.Modify('SolarBTM', Scaler=Existing[idx])
        NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, NewTech[idx])
        NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, NewTech[idx])
        DNG = Dispatch(NG)

        for jdx, Asset in enumerate(DNG.NG.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['Generation']/1000000/2)

    #Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.stackplot(NewTech*100, GenSum)
    #plt.legend(labels,bbox_to_anchor=(0, 1), loc='upper left', ncol=1, framealpha=1)
    return

def SweepSolarCarbon(NG, Start, Stop, Steps,Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    DynamScale = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)

    GenSum = np.ndarray(shape=(len(NG.Mix['Technologies']), len(Existing)))

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Grid.Load('Data/2016.NGM')
        NG = NG.Modify('Solar', Scaler=Existing[idx])
        NG = NG.Modify('SolarBTM', Scaler=Existing[idx])
        NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, NewTech[idx])
        NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, NewTech[idx])
        DNG = Dispatch(NG)

        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))

    # Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.stackplot(NewTech * 100, GenSum)
    #plt.legend(labels,bbox_to_anchor=(0, 1), loc='upper left', ncol=1, framealpha=1)
    return

def CarbonEmissions(NG, Start, Stop, Steps, Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    NG = NG.MatchDates()
    DynamScale = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)

    GenSum = np.ndarray(shape=(len(Existing)))

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Grid.Load('Data/2016.NGM')
        NG = NG.Modify('Solar', Scaler=Existing[idx])
        NG = NG.Modify('SolarBTM', Scaler=Existing[idx])
        NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, NewTech[idx])
        NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, NewTech[idx])
        DNG = Dispatch(NG)
        GenSum[idx] = np.sum(DNG.CarbonEmissions) / 2 * (1*10**-9)

    # Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.plot(NewTech*100, GenSum/GenSum[0] * 100)
    #plt.legend(labels)
    #plt.show()
    return

def MaxGenOfDay(NG,Tech,Enhancment):

    NG = NG.MatchDates()
    DynamScale = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)

    NG = Grid.Load('Data/2016.NGM')
    NG = NG.Modify('Solar',Scaler=0.5)
    NG = NG.Modify('SolarBTM',Scaler=0.5)
    NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, 0.5)
    NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, 0.5)
    NG = NG.MatchDates()
    DNG = Dispatch(NG)
    for Asset in DNG.Distributed.Mix['Technologies']:
        if Asset['Technology'] == Tech:
            #MaxGenTime = list()
            MaxGenTime = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].idxmax().hour for i in range(1,365)]
            MaxGenTimeMins = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].idxmax().minute for i in range(1, 365)]
            MaxGenTimeMins = [Min/60 for Min in MaxGenTimeMins]
            MaxGenTime = [a+b for a, b in zip(MaxGenTime,MaxGenTimeMins)]
            MaxGen = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].max() for i in range(1, 365)]
    plt.scatter(range(1,365),MaxGenTime)


    return

#NationalGrid = Grid("Mix2016.json")
#NationalGrid = NationalGrid.Add('SolarNT','Solar')
#NationalGrid = NationalGrid.Add('SolarBTMNT','SolarBTM')
#NationalGrid = NationalGrid.Save('Data','2016Raw')
NationalGrid = Grid.Load('Data/2016Raw.NGM')
NationalGrid = NationalGrid.Demand()
NationalGrid = NationalGrid.CarbonEmissions()
NationalGrid = NationalGrid.MatchDates()
#NationalGrid = NationalGrid.Add('SolarNT','Solar')
#NationalGrid = NationalGrid.Add('SolarBTMNT','SolarBTM')
#plt.figure(figsize=(6,8))
NationalGrid = NationalGrid.Save('Data','2016')
#SweepSolarCarbon(NationalGrid, 0, 1, 100,'Data/Devices/PolySi.csv')
#SweepSolarGen(NationalGrid, 0, 1, 100, 'Data/Devices/PolySi.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/PolySi.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/NewCastle.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/DSSC.csv')
MaxGenOfDay(NationalGrid,'Solar','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'Solar','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarNT','Data/Devices/NewCastle.csv')
plt.ylabel("Time of Max Generation")
plt.xlabel("Day of Year")
plt.show()
