import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

Data = pd.read_csv('Correlation2016.csv')
x = Data['Irradiance'].to_numpy()
y = Data['SolarGen'].to_numpy()
x = np.roll(x,1)
p = scipy.stats.pearsonr(x,y)
sr = scipy.stats.spearmanr(x,y)
tau = scipy.stats.kendalltau(x,y)

#m,b = np.polyfit(x,y,1)
#print(b)
plt.scatter(x,y)
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
#plt.plot(x,m*x+b,c='red')
plt.ylabel("Generation (MW)")
plt.xlabel("Irradiance (Wm$^{-2}$)")
print("Pearson: ", p)
print("Spearman: ", sr)
print("Kendall: ", tau)
plt.show()