import numpy as np

from Main import *


PCE0Max = 30
PCE0Sun = np.linspace(1,30,50)
PCE1Sun = np.linspace(1,30,50)
Carbon = np.zeros((len(PCE0Sun),len(PCE1Sun)))
EM = np.zeros((len(PCE0Sun),len(PCE1Sun)))

NG = 'Data/2016RawT.NGM'
lat = 53.13359
lon = -1.746826
NG = Grid.Load(NG)
NG.MatchDates()
NG.Demand()
NG.CarbonEmissions()
NG.PVGISFetchDefinedDeviceNOScaling(lat, lon)
for idx,x in enumerate(PCE1Sun):
    for idy,y in enumerate(PCE0Sun):
        EM[idx][idy] = x/y
        #if x/y < 1:
        #    Carbon[idx][idy] = np.nan
        #else:
        NG.DynamScaleLinear(x,y)
        NG = Scaling(NG, 1, 1, 0.5, 0.5)
        DNG = Dispatch(NG)
        #if idx == 0 and idy == 0:
            #Carbon0 = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))
        Carbon[idx][idy] = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))

X,Y = np.meshgrid(PCE1Sun,PCE0Sun)
plt.rcParams["figure.dpi"] = 300
plt.pcolormesh(X,Y,Carbon, cmap='inferno_r',vmax=100,vmin=40)
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(bottom=1)
plt.colorbar(label='Carbon Equivalent Emissions (Mt)')
cont = plt.contour(X,Y,EM,levels=[0.3,0.5,1,3,5],colors="white")
#cont2 = plt.contour(X,Y,Carbon,levels=[70],colors="g")
plt.clabel(cont, inline=True, fontsize=12, manual=[(5,25),(10,20),(15,15),(20,10),(25,5)])
#plt.clabel(cont2, inline=True, fontsize=12)
plt.xlabel('PCE Near 1 Suns (%)')
plt.ylabel('PCE Near 0 Suns (%)')
plt.show()