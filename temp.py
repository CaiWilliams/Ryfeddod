import pandas as pd


A = pd.read_csv('Data/Devices/DSSCPCE.csv')
print(A['Enhanced'].min()*100 )