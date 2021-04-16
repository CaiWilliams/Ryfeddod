import requests
import pandas as pd
from datetime import datetime, timedelta, date
import io

Key = 'zz6sqbg3mg0ybyc'
APIRequest = 'https://api.bmreports.com/BMRS/B1620/<V2>?APIKey=<zz6sqbg3mg0ybyc>&SettlementDate=<SettlementDate>&Period=<Period>&ServiceType=<xml/csv'
Year = 2020
NumDays = (datetime(Year, 12, 31) - datetime(Year, 1, 1)).days
Days = [datetime(Year, 1, 1) + timedelta(days=1 * Day) for Day in range(0, NumDays+1)]
DaysStr = [Day.strftime('%Y-%m-%d') for Day in Days]
AllAPIRequests = ['https://api.bmreports.com/BMRS/B1620/V1?APIKey=zz6sqbg3mg0ybyc&SettlementDate='+SettlementDate+'&Period=*&ServiceType=csv' for SettlementDate in DaysStr]
AllAPIAnswers = [requests.get(APIrequest) for APIrequest in AllAPIRequests]
ALLAPIDataframes = [pd.read_csv(io.StringIO(Answer.text), skiprows=[0, 1, 2, 3], skipfooter=1, engine='python', index_col=False).sort_values('Settlement Period') for Answer in AllAPIAnswers]
YearDataframe = pd.concat(ALLAPIDataframes, ignore_index=True)
YearDataframe = YearDataframe.drop(columns=['*Document Type', 'Business Type', 'Process Type', 'Time Series ID', 'Curve Type', 'Resolution', 'Active Flag', 'Document ID', 'Document RevNum'])
#YearDataframe = YearDataframe.set_index(['Settlement Date'])
#YearDataframe = pd.pivot_table(YearDataframe,values='Power System Resource  Type',index=['Settlement Date', 'Settlement Period'])
YearDataframe = YearDataframe.pivot_table(index=['Settlement Date','Settlement Period'], columns='Power System Resource  Type', values='Quantity')
YearDataframe.to_csv('Year.csv')

#r = requests.get(AllAPIRequests[0]).text
#df = pd.read_csv(io.StringIO(r), skiprows=[0, 1, 2, 3], skipfooter=1, engine='python')
