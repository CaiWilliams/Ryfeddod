import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit


def import_device(DeviceDir):
    df = pd.read_csv(DeviceDir)
    df = df.loc[~(df==0).all(axis=1)]
    df = df.drop_duplicates('Irradiance')
    return df


def func(x, a, b, c, d):
    return a * np.exp((-b * x)+d) + c

def func2(x, a, b, c):
    return a * np.exp(-b * x) + c

def interp2(Device):
    popt, pcov = curve_fit(func2, Device['Irradiance'], Device['Enhanced']*100)
    return popt

def interp(Device):
    popt, pcov = curve_fit(func, Device['Irradiance'], Device['Enhanced']*100)
    return popt

def Sigmoid(x, A, K, B, V, Q, C, x0):
    return A + ((K - A)/np.power((C + Q * np.exp(-B *(x-x0))), (1/V)))

def InterpSigmoid(Device):
    X = Device['Irradiance'].to_numpy()[:]
    Y = Device['Enhanced'].to_numpy()[:]
    popt, pcov = curve_fit(Sigmoid, X, Y, maxfev=10000000)
    return popt

Bangor = import_device('Data/Devices/DSSCPCE.csv')
#Newcastle = import_device('Data/Devices/NewCastlePCE.csv')

BangorF = InterpSigmoid(Bangor)
#NewcastleF = InterpSigmoid(Newcastle)

BangorIrr = Bangor['Irradiance'].to_numpy()
BangorPCE = Bangor['Enhanced'].to_numpy()*100

#NewcastleIrr = Newcastle['Irradiance'].to_numpy()
#NewcastlePCE = Newcastle['Enhanced'].to_numpy()

Xnew = np.linspace(-1000,1000,100)
A, K, B, V, Q, C, x0 = BangorF

plt.plot(BangorIrr, BangorPCE)

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0-50)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC-50.csv')

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0-0)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC-0.csv')

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0+50)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC+50.csv')

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0+100)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC+100.csv')

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0+150)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC+150.csv')

BangorFit = Sigmoid(Xnew, A, K, B, V, Q, C, x0+200)
plt.plot(Xnew, BangorFit*100)
Temp = pd.DataFrame()
Temp['Irradiance'] = Xnew
Temp['Enhanced'] = BangorFit*100
#Temp.to_csv('Data/Devices/DSSC+200.csv')



#plt.plot(NewcastleIrr, NewcastlePCE)
#plt.plot(Xnew, NewcastleFit/1000)
plt.xlabel("Irradiance (Wm$^{-2}$)")
plt.ylabel("Enhancement")
#plt.xlim(left=0,right=1000)

#plt.plot(NewcastleIrr, NewcastlePCE)
#plt.ylim(bottom=8,top=35)
#plt.plot(BangorIrr, BangorPCE)

#plt.plot(Xnew, BangorFitEnh, label='Bangor')
#plt.plot(Xnew, NewcastleFitEnh, label='Newcastle')


plt.show()