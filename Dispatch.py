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
        PVGISData['time'] = [t.replace(year=self.StartDate.year) for t in PVGISData['time']]
        PVGISData = PVGISData.set_index(['time'])
        PVGISData.index = PVGISData.index.rename('Settlement Date')

        IndexValues = [Asset['Generation'].index.values for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))
        PVGISData = PVGISData.loc[PVGISData.index.isin(CommonIndex)]
        self.PVGISData = PVGISData
        Enhancment = pd.read_csv(EnhancmentDir)
        f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
        self.DynamScale = f(PVGISData['G(i)'])
        self.DynamScalepd = PVGISData
        self.DynamScalepd['G(i)'] = self.DynamScale
        return self

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

def SweepSolarGen(NG, Start, Stop, Steps, Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
    print(DynamScale)

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

        SolarGen = 0
        for jdx, Asset in enumerate(DNG.NG.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['Generation']/1000000/2)
            if Asset['Technology'] == 'Fossil Gas' or Asset['Technology'] == 'Fossil Hard coal': #or Asset['Technology'] == 'SolarNT' or Asset['Technology'] == 'SolarBTMNT':
                SolarGen = SolarGen + np.sum(Asset['Generation']/1000000/2)
        print("Gas/Coal Generation (TWh) " + str(idx) + " :" + str(SolarGen))

    #Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.stackplot(NewTech*100, GenSum)
    #plt.legend(labels,bbox_to_anchor=(0, 1), loc='upper left', ncol=1, framealpha=1)
    return

def SweepSolarCarbon(NG, Start, Stop, Steps,Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
    print(DynamScale)

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

        C = 0
        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
            if Asset['Technology'] == 'Fossil Hard coal':
                C = C + np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
        print("Emissions (Mt) " + str(idx) + " :" + str(C))

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
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale

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
    plt.plot(NewTech*100, GenSum)
    #plt.legend(labels)
    #plt.show()
    return

def MaxGenOfDay(NG,Tech,Enhancment):

    NG = NG.MatchDates()
    NG = Grid.Load('Data/2016.NGM')
    NG = NG.Modify('Solar',Scaler=0.5)
    NG = NG.Modify('SolarBTM',Scaler=0.5)
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
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

            MaxSunTime = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].idxmax().hour for i in range(1, 365)]
            MaxSunTimeMins = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].idxmax().minute for i in range(1, 365)]
            MaxSunTimeMins = [Min / 60 for Min in MaxSunTimeMins]
            MaxSunTime = [a + b for a, b in zip(MaxSunTime, MaxSunTimeMins)]
            MaxSun = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].max() for i in range(1, 365)]

    #plt.scatter(range(1,365),MaxGenTime)
    plt.scatter(range(1,365),MaxSunTime)



    return

def DayIrradiance(NG,Enhancment, Month,Day):
    NG = Grid.Load('Data/2016.NGM')
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
    Month = NG.PVGISData.loc[NG.PVGISData.index.month == Month]
    Day = Month.loc[Month.index.day == Day]
    plt.plot(Day.index,Day['G(i)'])
    return Day

def AverageDayTechnologies(DNG,*args):

    for Technology in args:
        for Asset in DNG.NG.Mix['Technologies']:
            if Technology == Asset['Technology']:
                Means = Asset['Generation'].groupby(Asset['Generation'].index.hour).mean()
                plt.plot(Means)
    return

def AverageDayTechnologiesMonth(DNG, Month, *args):
    for Technology in args:
        for Asset in DNG.NG.Mix['Technologies']:
            if Technology == Asset['Technology']:
                Means = Asset['Generation'].loc[Asset['Generation'].index.month == Month]
                Means = Means.groupby(Means.index.hour).mean()
                plt.plot(Means)
    return


def SolarNoon(NG,Lat,Lon):
    StartDate = NG.StartDate
    EndDate = NG.EndDate
    NumDays = (EndDate - StartDate).days
    Days = [StartDate + timedelta(days=1 * Day) for Day in range(0, NumDays + 1)]
    DaysStr = [Day.strftime('%Y-%m-%d') for Day in Days]
    AllAPIRequests = ['https://api.sunrise-sunset.org/json?lat='+str(Lat)+'&lng='+str(Lon)+'&date='+Day for Day in DaysStr]
    AllAPIAnswers = [requests.get(APIrequest).json() for APIrequest in AllAPIRequests]
    #ALLAPISolarNoon = [APIANfor APIAnswer in AllAPIAnswers]
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
#plt.figure(figsize=(8,12))
#plt.rcParams.update({'font.size': 24})
#NationalGrid = NationalGrid.Save('Data','2016')
#SweepSolarCarbon(NationalGrid, 0, 1, 100,'Data/Devices/NewCastle.csv')
SweepSolarGen(NationalGrid, 0, 1, 100, 'Data/Devices/DSSC.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/PolySi.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/NewCastle.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarBTMNT','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarBTMNT','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarNT','Data/Devices/NewCastle.csv')
#DayIrradiance(NationalGrid, 'Data/Devices/DSSC.csv',6,30).to_csv('20160630Irradiance.csv')


#NationalGrid = NationalGrid.PVGISFetch('Data/Devices/DSSC.csv', 53.13359, -1.746826)
#DynamScale = NationalGrid.DynamScale
#plt.plot(NationalGrid.PVGISData,c='tab:orange')
#plt.twinx()
#plt.plot(NationalGrid.PVGISData.index,DynamScale)

#NationalGrid = NationalGrid.Modify('Solar', Scaler=0.5)
#NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=0.5)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarNT', DynamScale, 0.5)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, 0.5)
#DNG = Dispatch(NationalGrid)

#AverageDayTechnologiesMonth(DNG,1,'Solar','SolarNT')
#plt.xlabel("Hour of the day")
#plt.ylabel("Mean Gneeration (MW)")
#plt.twinx()
#A = NationalGrid.PVGISData.loc[NationalGrid.PVGISData.index.month == 1]
#A = A.replace(0, np.NaN)#
#A = A.groupby(A.index.hour).mean()
#plt.plot(A.index, A)
#A = NationalGrid.PVGISData.loc[NationalGrid.PVGISData.index.month == 7]
#A = A.replace(0, np.NaN)
#A = A.groupby(A.index.hour).mean()
#plt.plot(A.index, A)
#plt.ylabel("Mean Irradiance (Wm$^{-2}$)")
#plt.ylabel("Carbon Equivalent Emissions (Mt)")
#plt.xlabel("Proportion of DAPV in Grid (%)")
plt.show()
