import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

class Original:

    def __init__(self,FilePath,Country):
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

class Technology(Original):

    def __init__(self,OG,Tech):
        super().__init__(OG.FilePath,OG.Country)
        Data = self.Data[self.Data['Fuel_Code'] == Tech]
        self.Total = Data['Generation'].sum(axis=0)
        DataNS = Data.drop(['Site_Code','POC_Code','Nwk_Code','Gen_Code','Fuel_Code','Tech_Code'],axis=1).reset_index()
        self.HourlyTotal = DataNS.groupby(DataNS['Trading_date']).agg({'Generation':'sum'})
        self.DailyTotal = DataNS.groupby(DataNS['Trading_date'].dt.date).agg({'Generation':'sum'})
        self.MonthlyTotal = DataNS.groupby(DataNS['Trading_date'].dt.month).agg({'Generation':'sum'})
        self.Raw = self.Data[self.Data['Fuel_Code'] == Tech]
    
    def __rmul__(self,other):
        return self.Total * other
    

class DispatchClass:

    def __init__(self,*args):
        self.Technologies = np.empty(0)
        for i in args:
            self.Technologies = np.append(self.Technologies,i)
        self.Total = 0
        self.HourlyTotal = args[0].HourlyTotal
        self.HourlyTotal['Generation'] = 0
        for i in self.Technologies:
            self.Total = self.Total + i.Total
            self.HourlyTotal = self.HourlyTotal.add(i.HourlyTotal)
        self.HourlyTotal = self.HourlyTotal.dropna()


NZ = Original('Data','NZ')

Hydro = Technology(NZ,'Hydro')
Wind = Technology(NZ,'Wind')
Gas = Technology(NZ,'Gas')
Coal = Technology(NZ,'Coal')
Geo = Technology(NZ,'Geo')
Wood = Technology(NZ,'Wood')
Diesel = Technology(NZ,'Diesel')

DC1 = DispatchClass(Hydro,Wind,Geo)
DC2 = DispatchClass(Wood,Gas)
DC3 = DispatchClass(Coal,Diesel)

plt.stackplot(DC1.HourlyTotal.index,DC1.HourlyTotal['Generation'],DC2.HourlyTotal['Generation'],DC3.HourlyTotal['Generation'],labels=['DC1','DC2','DC3'])
plt.legend()
plt.show()
