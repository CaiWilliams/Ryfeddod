import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Bangor = pd.read_csv('Data/Devices/DSSC.csv')
Newcastle = pd.read_csv('Data/Devices/NewCastle.csv')
Bangor_initial_PCE = 0.0078
Newcastle_initial_PCE = 0.09875

plt.plot(Bangor['Irradiance'] ,Bangor['Irradiance']*Bangor['Enhanced'])
plt.plot(Newcastle['Irradiance'] ,Newcastle['Irradiance']*Newcastle['Enhanced'])
plt.show()

print(np.sum(Bangor['Irradiance']*Bangor['Enhanced']))

Bangor_Eh = Bangor['Enhanced'].to_numpy()
Bangor_Irr = Bangor['Irradiance'].to_numpy()

