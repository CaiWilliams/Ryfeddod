import pandas as pd
from datetime import datetime, timedelta

dt = pd.read_csv('Data/UK2/Year.csv')
dt['Settlement Period'] = [timedelta(minutes=int(Period*30)) for Period in dt['Settlement Period']]
dt['Settlement Date'] = pd.to_datetime(dt['Settlement Date'], format='%Y-%m-%d')
dt['Settlement Date'] = dt['Settlement Date'] + dt['Settlement Period']
dt = dt.drop(columns='Settlement Period')
print(dt)