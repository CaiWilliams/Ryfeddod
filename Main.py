import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import json
import pickle
import pytz
import requests
import io
#from pvlive_api import PVLive
from scipy.interpolate import interp1d
import copy
import xml.etree.ElementTree as et
import os

class Grid:

    #Initialises and fetches the data from the assigned data sources form the mix_dir file
    def __init__(self, mix_dir):
        self.BMRSKey = "zz6sqbg3mg0ybyc" # To run please obtain an API key for BMRS
        self.ENTSOEKey = "6f7dd5a8-ca23-4f93-80d8-0c6e27533811" # To run please obtain an API key for ENTSOE

        with open(mix_dir) as Mix_File:
            self.Mix = json.load(Mix_File)

        DataSources = set()
        for Tech in self.Mix["Technologies"]:
            DataSources.add(Tech["Source"])

        self.StartDate = datetime.strptime(self.Mix['StartDate'], '%Y-%m-%d')
        self.EndDate = datetime.strptime(self.Mix['EndDate'], '%Y-%m-%d')
        self.timezone = 'Europe/London'

        if "BMRS" in DataSources:
            self.bmrs_fetch()
        if "PVLive" in DataSources:
            self.pv_live_fetch()

        for Tech in self.Mix['Technologies']:
            if Tech['Source'] == "BMRS":
                Tech['Generation'] = self.BMRSData[str(Tech['Technology'])]
                Tech['Generation'] = Tech['Generation'].rename(
                    'Generation')
            if Tech['Source'] == "PVLive":
                Tech['Generation'] = self.PVLiveData['generation_mw']
                Tech['Generation'] = Tech['Generation'].rename('Generation')

        if "ENTSOE" in DataSources:
            self.timezone = self.Mix['Timezone']
            self.domain = self.Mix['Domain']
            self.entsoe_fetch()

        for Tech in self.Mix['Technologies']:
            if Tech['Source'] == 'ENTSOE':
                Tech['Generation'] = self.ENTSOEData[str(Tech['Technology'])]
                Tech['Generation'] = Tech['Generation'].rename('Generation')
    # Converts Mix file start and tend period to match ENTSOE format
    def convert_period_format(self, date_obj, timezone):
        timezone = pytz.timezone(timezone)
        date_obj = timezone.localize(date_obj)
        date_obj = date_obj.astimezone(pytz.utc)
        api_format = date_obj.strftime('%Y%m%d%H%M')
        return api_format

    # Converts county name to ENTESOE Code
    def entsoe_codes(self):
        E = pd.read_csv('Data/ENTSOELocations.csv')
        N = ['Name 0', 'Name 1', 'Name 2']

        for X in N:
            EX = E[X].dropna().to_list()
            DN = [string for string in EX if self.domain in string]
            if len(DN) > 0:
                break

        Code = E[E[X].isin([DN[0]])]['Code'].values[0]
        return Code

    # Fetches aggregated generation data for the specified domain and time from ENTSOE
    def aggregated_generation(self, start_period, end_period):
        base = 'https://web-api.tp.entsoe.eu/api?'
        security_token = 'securityToken=' + str(self.ENTSOEKey)
        document_type = 'documentType=A75'
        process_type = 'processType=A16'
        in_domain = 'in_domain=' + str(self.entsoe_codes())
        period_start = 'periodStart=' + str(start_period)
        period_end = 'periodEnd=' + str(end_period)
        api_call = base + security_token + '&' + document_type + '&' + process_type + '&' + in_domain + '&' + period_start + '&' + period_end
        api_answer = requests.get(api_call)
        return api_answer

    # Converts time resolution reported by ENTSOE to timedelta type
    def time_res_to_delta(self, res):
        res = res.replace('PT', '')
        res = res.replace('M', '')
        res = float(res)
        return timedelta(minutes=res)

    # Converts position in ENTSOE data to time
    def position_to_time(self, pos, res, start):
        return [start + (res * x) for x in pos]

    # Converts ENTSOE generation codes to asset type text
    def type_code_to_text(self, asset_type):
        codes = pd.read_csv('Data/EUPsrType.csv')
        return codes[codes['Code'] == asset_type]['Meaning'].values[0]

    # Matches dates in ENTSOE Data
    def match_dates_entsoe(self, dic):
        index_values = [dic[x][0] for x in dic.keys()]
        common_index_values = list(set.intersection(*map(set, index_values)))
        for x in dic.keys():
            times = np.unique(dic[x][0])
            mask = np.in1d(times, common_index_values)
            mask = np.where(mask)[0]
            dic[x] = dic[x][:, mask]
        return dic

    # Fetches and formats the aggregated generation from ENTSOE
    def entsoe_fetch(self):

        start_period = self.convert_period_format(self.StartDate, self.timezone)
        end_period = self.convert_period_format(self.EndDate, self.timezone)
        data = self.aggregated_generation(start_period, end_period)
        root = et.fromstring(data.content)
        ns = {'d': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
        entsoe_data = {}

        for GenAsset in root.findall('d:TimeSeries', ns):
            asset_type = GenAsset.find('d:MktPSRType', ns)
            asset_type = asset_type.find('d:psrType', ns).text
            asset_type = self.type_code_to_text(asset_type)
            data = GenAsset.find('d:Period', ns)
            dates = data.find('d:timeInterval', ns)
            start_date = datetime.strptime(dates.find('d:start', ns).text, '%Y-%m-%dT%H:%MZ')
            resolution = data.find('d:resolution', ns).text
            resolution = self.time_res_to_delta(resolution)
            generation = data.findall('d:Point', ns)
            time = [float(x.find('d:position', ns).text) for x in generation]
            time = self.position_to_time(time, resolution, start_date)
            generation = [float(x.find('d:quantity', ns).text) for x in generation]
            tmp = np.vstack((time, generation))

            if asset_type in entsoe_data:
                tmp2 = entsoe_data.get(asset_type)
                tmp2 = np.hstack((tmp2, tmp))
                entsoe_data[asset_type] = tmp2
            else:
                entsoe_data[asset_type] = tmp

        entsoe_data = self.match_dates_entsoe(entsoe_data)
        entsoe_data_pd = pd.DataFrame()
        for asset in entsoe_data.keys():
            entsoe_data_pd[asset] = entsoe_data[asset][1]
            entsoe_data_pd['Settlement Date'] = entsoe_data[asset][0]

        self.Dates = entsoe_data_pd['Settlement Date']
        entsoe_data_pd = entsoe_data_pd.set_index('Settlement Date')
        entsoe_data_pd = entsoe_data_pd.fillna(0)
        entsoe_data_pd.index = entsoe_data_pd.index.tz_localize('UTC').tz_convert(self.timezone)
        self.ENTSOEData = entsoe_data_pd

        return self.ENTSOEData

    # Fetches and formats BMRS data
    def bmrs_fetch(self):

        NumDays = (self.EndDate - self.StartDate).days
        Days = [self.StartDate + timedelta(days=1 * Day) for Day in range(0, NumDays + 1)]
        DaysStr = [Day.strftime('%Y-%m-%d') for Day in Days]

        AllAPIRequests = ['https://api.bmreports.com/BMRS/B1620/V1?APIKey=' + self.BMRSKey + '&SettlementDate=' + SettlementDate + '&Period=*&ServiceType=csv' for SettlementDate in DaysStr]
        AllAPIAnswers = [requests.get(APIrequest) for APIrequest in AllAPIRequests]
        ALLAPIDataframes = [pd.read_csv(io.StringIO(Answer.text), skiprows=[0, 1, 2, 3], skipfooter=1, engine='python', index_col=False).sort_values('Settlement Period') for Answer in AllAPIAnswers]

        YearDataframe = pd.concat(ALLAPIDataframes, ignore_index=True)
        YearDataframe = YearDataframe.drop(columns=['*Document Type', 'Business Type', 'Process Type', 'Time Series ID', 'Curve Type', 'Resolution', 'Active Flag', 'Document ID', 'Document RevNum'])
        YearDataframe = YearDataframe.pivot_table(index=['Settlement Date', 'Settlement Period'], columns='Power System Resource  Type', values='Quantity')
        YearDataframe = YearDataframe.reset_index()

        YearDataframe["Settlement Period"] = [timedelta(minutes=int(Period * 30)) for Period in YearDataframe['Settlement Period']]
        YearDataframe['Settlement Date'] = pd.to_datetime(YearDataframe['Settlement Date'], format='%Y-%m-%d')
        YearDataframe['Settlement Date'] = YearDataframe['Settlement Date'] + YearDataframe['Settlement Period']
        YearDataframe = YearDataframe.drop(columns=['Settlement Period'])

        timezone = pytz.timezone('Europe/London')
        YearDataframe['Settlement Date'] = [t.replace(tzinfo=timezone) for t in YearDataframe['Settlement Date']]
        YearDataframe['Settlement Date'] = [t.astimezone(pytz.utc) for t in YearDataframe['Settlement Date']]

        self.Dates = YearDataframe['Settlement Date']
        YearDataframe = YearDataframe.set_index("Settlement Date")
        YearDataframe = YearDataframe.fillna(0)

        self.BMRSData = YearDataframe
        return self.BMRSData

    # Fetches and formats PV Live data
    def pv_live_fetch(self):

        pvl = PVLive()

        tz2 = pytz.timezone('Europe/London')
        self.StartDate = tz2.localize(self.StartDate)
        self.EndDate = tz2.localize(self.EndDate)
        self.PVLiveData = pvl.between(self.StartDate, self.EndDate, dataframe=True)

        self.PVLiveData = self.PVLiveData.sort_values(by=['datetime_gmt'])
        self.PVLiveData = self.PVLiveData.set_index('datetime_gmt')
        self.PVLiveData = self.PVLiveData.fillna(0)

        self.PVLiveData.index = self.PVLiveData.index.rename('Settlement Date')
        self.PVLiveData.to_csv("BTM.csv")

        return self.PVLiveData

    # Adds a copy of an existing technology in mix
    def add(self, name, tech, scaler):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == tech:
                Asset_Copy = copy.deepcopy(Asset)
                Asset_Copy['Technology'] = name
                Asset_Copy['Scaler'] = scaler
                self.Mix['Technologies'].append(Asset_Copy)
                return self

    # Modifies a property of a technology defined in the mix
    def modify(self, tech, **kwags):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == tech:
                Asset.update(kwags)
        return self

    # Fetches and formats PVGIS data
    def pvgis_fetch(self, enhancment_dir, latitude, longitude):

        Startyear = self.StartDate.year
        EndYear = self.EndDate.year
        PVGISAPICall = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=" + str(latitude) + "&lon=" + str(longitude) + "&startyear=" + str(Startyear) + "&endyear=" + str(EndYear) + "&outputformat=csv&optimalinclination=1&optimalangles=1"
        PVGISAnswer = requests.get(PVGISAPICall)

        PVGISData = pd.read_csv(io.StringIO(PVGISAnswer.text), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
        PVGISData['time'] = pd.to_datetime(PVGISData['time'], format='%Y%m%d:%H%M')
        PVGISData['time'] = [t.replace(minute=0) for t in PVGISData['time']]

        GHalfHours = PVGISData['G(i)'].to_numpy()
        GHalfHours = np.insert(GHalfHours, -1, 0)
        GHalfHours = (GHalfHours[1:] + GHalfHours[:-1]) / 2
        THalfHours = PVGISData['time'] + timedelta(minutes=30)
        THalfHours = THalfHours.iloc[:]
        HalfHours = pd.DataFrame(THalfHours)
        HalfHours['G(i)'] = GHalfHours
        PVGISData = pd.concat([PVGISData, HalfHours])
        PVGISData = PVGISData.sort_values(by=['time'])
        PVGISData['time'] = [t.replace(year=self.StartDate.year) for t in PVGISData['time']]
        utc = tz.gettz('UTC')
        timezone = tz.gettz(self.timezone)
        #PVGISData['time'] = [t.replace(tzinfo=utc) for t in PVGISData['time']]
        #PVGISData['time'] = [t.astimezone(timezone) for t in PVGISData['time']]
        #PVGISData['time'] = PVGISData['time'].tz_localize('UTC').tz_convert(self.timezone)
        PVGISData = PVGISData.set_index(['time'])
        PVGISData.index = PVGISData.index.tz_localize('UTC').tz_convert(self.timezone)
        PVGISData.index = PVGISData.index.rename('Settlement Date')
        PVGISData.index = PVGISData.index

        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))
        PVGISData = PVGISData.loc[PVGISData.index.isin(CommonIndex)]

        self.PVGISData = copy.deepcopy(PVGISData)
        self.match_dates_to_pvgis()
        Enhancment = pd.read_csv(enhancment_dir)
        f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(), fill_value="extrapolate")
        self.DynamScale = f(self.PVGISData['G(i)'])
        return self

    # Match dates from each data source to each other
    def match_dates(self):

        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))

        Lengths = np.zeros(len(self.Mix['Technologies']))
        for idx, Asset in enumerate(self.Mix['Technologies']):
            Asset['Generation'] = Asset['Generation'].loc[Asset['Generation'].index.isin(CommonIndex)]
            Asset['Generation'] = Asset['Generation'][~Asset['Generation'].index.duplicated(keep='first')]
            self.Dates = Asset['Generation'].index
            Lengths[idx] = len(Asset['Generation'].index)

        return self

    # Match dates from PVGIS to other sources
    def match_dates_to_pvgis(self):
        CommonIndex = self.PVGISData.index.tolist()

        Lengths = np.zeros(len(self.Mix['Technologies']))
        for idx, Asset in enumerate(self.Mix['Technologies']):
            Asset['Generation'] = Asset['Generation'].loc[Asset['Generation'].index.isin(CommonIndex)]
            Asset['Generation'] = Asset['Generation'][~Asset['Generation'].index.duplicated(keep='first')]
            self.Dates = Asset['Generation'].index
            Lengths[idx] = len(Asset['Generation'].index)

        return self

    # Calculates the historical demand (assuming historical generation perfectly met demand)
    def demand(self):
        self.Demand = pd.DataFrame(index=self.Mix['Technologies'][0]['Generation'].index.copy())
        self.Demand = 0
        for Asset in self.Mix['Technologies']:
            self.Demand = self.Demand + Asset['Generation'][:]
        return self

    # Calculate the historical carbon emissions
    def carbon_emissions(self):
        self.CarbonEmissions = pd.DataFrame(index=self.Mix['Technologies'][0]['Generation'].index.copy())
        self.CarbonEmissions = 0

        for Asset in self.Mix['Technologies']:
            self.CarbonEmissions = self.CarbonEmissions + (Asset['Generation'][:] * Asset['CarbonIntensity'])
        return self

    # Saves the class object as a pickle fine (.NGM [National Grid Mix])
    def save(self, f_dir, filename):
        with open(os.path.join(os.getcwd(),f_dir,str(filename) + '.NGM'), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    # Loads a class object to be further operated on
    def load(f_dir):
        with open(f_dir, 'rb') as handle:
            return pickle.load(handle)


class Dispatch:

    # Initialises and dispatches the grid object given
    def __init__(self, ng):
        self.NG = copy.deepcopy(ng)
        self.Original = copy.deepcopy(ng)
        self.Distributed = copy.deepcopy(ng)
        self.Demand = self.Distributed.Demand
        self.Generation = self.Distributed.Demand
        self.Generation = 0
        self.CarbonEmissions = self.Distributed.Demand
        self.CarbonEmissions = 0

    def run(self, ss, snts):

        self.scaling(ss, snts)

        self.set_dispatch_class('Hydro Pumped Storage', 2)

        self.order()

        self.distribute(self.DC1)
        self.distribute(self.DC2)

        self.set_scaler('Hydro Pumped Storage', 0)

        #self.distribute(self.DC3)

        self.storage()
        self.distribute(self.DC4)
        self.undersupply()
        self.misc()

        return self

    # Sets the scaling the for traditional solar and new technology solar
    def scaling(self, solar_scaler, solar_nt_scaler):
        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['Technology'] == 'Solar':
                Asset['Scaler'] = solar_scaler
            if Asset['Technology'] == 'SolarNT':
                Asset['Scaler'] = self.NG.DynamScale * solar_nt_scaler

    def DynamicScalingFromFile(self,Tech, Dir, BaseScale):
        DynamScaler = pd.read_csv(Dir, parse_dates=['T'], index_col=['T'])
        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))#

        DynamScaler = DynamScaler[DynamScaler.index.isin(CommonIndex)]
        DynamScaler = DynamScaler['Enhancment'].to_numpy()[1:-1]
        Scale = DynamScaler * BaseScale

        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset['Scaler'] = Scale[:]

        return self

    def set_scaler(self,name,scaler):
        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['Technology'] == name:
                Asset['Scaler'] = scaler
        return self

    def set_dispatch_class(self, name, dispatch_class):
        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['Technology'] == name:
                Asset['DispatchClass'] = dispatch_class

    # Sets the order which each dispatch class will be distributed (Order writen in .json file)
    def order(self):

        self.DC1 = np.zeros(0)
        self.DC2 = np.zeros(0)
        self.DC3 = np.zeros(0)
        self.DC4 = np.zeros(0)

        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['DispatchClass'] == 4:
                self.DC4 = np.append(self.DC4, Asset)
            elif Asset['DispatchClass'] == 3:
                self.DC3 = np.append(self.DC3, Asset)
            elif Asset['DispatchClass'] == 2:
                self.DC2 = np.append(self.DC2, Asset)
            elif Asset['DispatchClass'] == 1:
                self.DC1 = np.append(self.DC1, Asset)
        return

    # Distributes the given dispatch class
    def distribute(self, dc):
        for Asset in dc:

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

    # if under supply occurs fill gap with DC4 assets
    def undersupply(self):
        if np.any((self.Demand - self.Generation)):
            for Asset in self.DC4:
                MaxGen = Asset['Capacity']
                DemandRemaning = self.Demand - self.Generation
                Gen = np.minimum(MaxGen, DemandRemaning)
                self.Generation = self.Generation + Gen
                self.CarbonEmissions = self.CarbonEmissions + (Gen * Asset['CarbonIntensity'])
                for DissributedAsset in self.Distributed.Mix['Technologies']:
                    if Asset['Technology'] == DissributedAsset['Technology']:
                        DissributedAsset['Generation'] = DissributedAsset['Generation'] + Gen
                        DissributedAsset['CarbonEmissions'] = DissributedAsset['CarbonEmissions'] + (Gen * Asset['CarbonIntensity'])
        return

    # Storage model
    def storage(self):

        StorageRTE = 0.92
        DemandRemaining = self.Demand - self.Generation

        StorageCapacity = 0
        StorageIntensity = 0
        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['Technology'] == "Hydro Pumped Storage":
                StorageCapacity = Asset['Capacity']
                StorageIntensity = Asset['CarbonIntensity']
                self.NormalStorage = np.minimum(Asset['Generation'], DemandRemaining)

        Pre = 0
        Post = 0
        StoragePower = StorageCapacity

        for Asset in self.DC2:
            for AssetPre in self.NG.Mix['Technologies']:
                if Asset['Technology'] == AssetPre['Technology']:
                    Pre += np.ravel((AssetPre['Generation'] * AssetPre['Scaler']).to_numpy(na_value=0))

            for AssetPost in self.Distributed.Mix['Technologies']:
                if Asset['Technology'] == AssetPost['Technology']:
                    Post += np.ravel((AssetPost['Generation'] * AssetPost['Scaler']).to_numpy(na_value=0))

        self.Pre = Pre
        self.Post = Post
        self.DC2Curtailed = Post - Pre

        self.StorageSOC = np.zeros(len(self.DC2Curtailed))
        self.StorageDischarge = np.zeros(len(self.DC2Curtailed))

        for idx in range(len(self.DC2Curtailed)):
            if idx == 0:
                self.StorageSOC[idx] = 0
                self.StorageDischarge[idx] = min(DemandRemaining[idx], self.StorageSOC[idx] * StorageRTE, StorageCapacity)
            else:
                self.StorageSOC[idx] = min(self.StorageSOC[idx - 1] + (self.DC2Curtailed[idx] * StorageRTE) - (self.StorageDischarge[idx] / StorageRTE), StoragePower)
                self.StorageDischarge[idx] = min(DemandRemaining[idx], self.StorageSOC[idx - 1] * StorageRTE, StorageCapacity)
                self.StorageSOC[idx] = min(self.StorageSOC[idx - 1] + (self.DC2Curtailed[idx] * StorageRTE) - (self.StorageDischarge[idx] / StorageRTE), StoragePower)
                self.StorageDischarge[idx] = min(DemandRemaining[idx], self.StorageSOC[idx - 1] * StorageRTE, StorageCapacity)

        self.Generation += self.StorageDischarge
        self.StorageCO2 = self.StorageDischarge * StorageIntensity
        self.CarbonEmissions = self.CarbonEmissions + self.StorageCO2

        for DissributedAsset in self.Distributed.Mix['Technologies']:
            if DissributedAsset['Technology'] == "Hydro Pumped Storage":
                DissributedAsset['Generation'] += self.StorageDischarge
                DissributedAsset['CarbonEmissions'] = DissributedAsset['CarbonEmissions'] + (self.StorageDischarge * DissributedAsset['CarbonIntensity'])

        return

    # Calculates oversupply and errors
    def misc(self):
        self.Oversuply = self.Generation - self.Demand
        self.Error = np.where(self.Oversuply != 0, True, False)


def setup(ng, device, lat, lon):
    ng = Grid.load(ng)
    ng.pvgis_fetch(device, lat, lon)
    ng.match_dates()
    ng.demand()
    ng.carbon_emissions()
    return ng


def create_file(mix_dir, file_name):
    NG = Grid(mix_dir)
    NG.match_dates()
    NG.add('SolarNT','Solar',0)
    NG.save('Data', file_name)
    return


if __name__ == "__main__":
    #create_file('Mix2016DE.json', '2016DE')
    #create_file('Mix2016CZ.json', '2016CZ')
    #create_file('Mix2016FR.json', '2016FR')
    create_file('Mix2016GB.json', '2016GB')