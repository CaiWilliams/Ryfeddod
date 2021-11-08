import requests
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from dateutil import tz
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

def Fetch(Latitude, Longitude):
    Startyear = 2016
    EndYear = 2016
    PVGISAPICall = "https://re.jrc.ec.europa.eu/api/seriescalc?lat=" + str(Latitude) + "&lon=" + str(
        Longitude) + "&startyear=" + str(Startyear) + "&endyear=" + str(
        EndYear) + "&outputformat=csv&optimalinclination=1&optimalangles=1"
    PVGISAnswer = requests.get(PVGISAPICall)
    if PVGISAnswer.status_code != 200:
        print("Error: ", PVGISAnswer.status_code, " Lat: ", Latitude, " Lon: ", Longitude)
        time.sleep(2)
        Fetch(Latitude,Longitude)
    else:
        return PVGISAnswer

def PVGISFetch(Latitude, Longitude):
    Startyear = 2016
    EndYear = 2016
    PVGISAnswer = Fetch(Latitude,Longitude)
    PVGISData = pd.read_csv(io.StringIO(PVGISAnswer.text), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7],
                            engine='python', usecols=['time', 'G(i)'])
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

    return PVGISData['G(i)'].to_numpy()

def PVGISFetchTime(Latitude, Longitude):
    Startyear = 2016
    EndYear = 2016
    PVGISAPICall = "https://re.jrc.ec.europa.eu/api/seriescalc?lat=" + str(Latitude) + "&lon=" + str(
        Longitude) + "&startyear=" + str(Startyear) + "&endyear=" + str(
        EndYear) + "&outputformat=csv&optimalinclination=1&optimalangles=1"
    PVGISAnswer = requests.get(PVGISAPICall)
    PVGISData = pd.read_csv(io.StringIO(PVGISAnswer.text), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7],
                            engine='python', usecols=['time', 'G(i)'])
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
    PVGISData['time'] = [t.replace(year=Startyear) for t in PVGISData['time']]
    # PVGISData['time'] = [t + timedelta(minutes=60) for t in PVGISData['time']]
    utc = tz.gettz('UTC')
    timezone = tz.gettz('Europe/London')
    PVGISData['time'] = [t.replace(tzinfo=utc) for t in PVGISData['time']]
    PVGISData['time'] = [t.astimezone(timezone) for t in PVGISData['time']]
    # PVGISData['time'] = [t + timedelta(minutes=30) for t in PVGISData['time']]
    PVGISData = PVGISData.set_index(['time'])
    PVGISData.index = PVGISData.index.rename('Settlement Date')

    return PVGISData.index.to_numpy()

def Enhance(Data, Enhancmentdir):
    Enhancment = pd.read_csv(Enhancmentdir)
    f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(), kind='slinear',fill_value="extrapolate")
    return f(Data)

locations = pd.read_csv('RandomLocs100.csv')
locations = locations
latitudes = locations['Latitude'].to_numpy()
longitudes = locations['Longitude'].to_numpy()
Cords = zip(latitudes,longitudes)


Time = PVGISFetchTime(latitudes[0], longitudes[0])
Data = [PVGISFetch(lat,lon) for lat,lon in Cords]


#Data = [Enhance(x,'Data/Devices/DSSC.csv') for x in Data]

Data = np.sum(Data, axis=0) / len(Data)

D = pd.DataFrame()
D.index = Time
D['Irradiance'] = Data
D.to_csv('100LocationIrradiance.csv')


