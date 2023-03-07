from Main import *
import matplotlib.pyplot as plt
import os


def match_dates(Gen,Price):
    IndexValues = [Gen.index, Price.index]
    CommonIndex = list(set.intersection(*map(set, IndexValues)))

    Gen = Gen.loc[Gen.index.isin(CommonIndex)]
    Gen = Gen[~Gen.index.duplicated(keep='first')]

    Price = Price.loc[Price.index.isin(CommonIndex)]
    Price = Price[~Price.index.duplicated(keep='first')]

    return Gen,Price

market_index_data = pd.read_csv('C:\\Users\cai_w\Documents\Programming\Ryfeddod\Data\\2016_MID.csv', delimiter=',')

market_index_data["Settlement Period"] = [timedelta(minutes=int(Period * 30)) for Period in market_index_data['Settlement Period']]
market_index_data['Settlement Date'] = pd.to_datetime(market_index_data['Settlement Date'], format='%d-%b-%Y')
market_index_data['Settlement Date'] = market_index_data['Settlement Date'] + market_index_data['Settlement Period']
market_index_data['Settlement Date'] = market_index_data['Settlement Date'].dt.tz_localize('UTC')

market_index_data = market_index_data[market_index_data['Market Index Data Provider Id'] == 'APXMIDP']
market_index_data = market_index_data.drop(columns=['Settlement Period', 'Market Index Data Provider Id'])
market_index_data = market_index_data.set_index('Settlement Date')
Price = market_index_data['Market Index Price(£/MWh)'] * 1.356


#ng = 'C:\\Users\cai_w\Documents\Programming\Ryfeddod\Data\\NationalGrid_2016.NGM'
ng = os.path.join(os.path.dirname(os.getcwd()),'Data','NationalGrid_2016.NGM')
#device = 'C:\\Users\cai_w\Documents\Programming\Ryfeddod\Data\Devices\\Newcastle48U.csv'
device = os.path.join(os.path.dirname(os.getcwd()),'Data','Devices','Newcastle48U.csv')
lat = 53.13359
lon = -1.746826

ng = Grid.load(ng)
ng = ng.add('5MWSolar', 'Solar', 0)
ng = ng.add('5MWSolarNT', 'SolarNT', 0)
ng.pvgis_fetch(device, lat, lon)
ng.match_dates()
ng.demand()
ng.carbon_emissions()


solar_capacity = 0
for asset in ng.Mix['Technologies']:
    if asset['Technology'] == 'Solar':
        solar_capacity = asset['Capacity']


farm_scaler = 5/solar_capacity

dng = Dispatch(ng)
dng = dng.set_scaler('5MWSolarNT',0)#farm_scaler * ng.DynamScale)
dng = dng.set_scaler('5MWSolar',farm_scaler/2)
dng = dng.run(1,0)

Gen = 0
for asset in dng.Distributed.Mix['Technologies']:
    if asset['Technology'] == '5MWSolar':
        Gen += asset['Generation']
    if asset['Technology'] == '5MWSolarNT':
        Gen += asset['Generation']

#print(dng.Original.CarbonEmissions[-1]/1000)
#print((dng.CarbonEmissions[-1]/1000)-(dng.Original.CarbonEmissions[-1]/1000))
Gen,Price = match_dates(Gen, Price)
solar_rev = Gen*Price

dng = Dispatch(ng)
dng = dng.set_scaler('5MWSolarNT',(farm_scaler * ng.DynamScale)/2)
dng = dng.set_scaler('5MWSolar', 0)#farm_scaler/2)#farm_scaler)
dng = dng.run(0,1)

Gen = 0
for asset in dng.Distributed.Mix['Technologies']:
    if asset['Technology'] == '5MWSolar':
        Gen += asset['Generation']
    if asset['Technology'] == '5MWSolarNT':
        Gen += asset['Generation']

Gen,Price = match_dates(Gen, Price)
solarNT_rev = Gen * Price

plt.plot(np.cumsum(solar_rev))
plt.plot(np.cumsum(solarNT_rev))

ng = os.path.join(os.path.dirname(os.getcwd()),'Data','NationalGrid_2016.NGM')
#device = 'C:\\Users\cai_w\Documents\Programming\Ryfeddod\Data\Devices\\Newcastle48U.csv'
device = os.path.join(os.path.dirname(os.getcwd()),'Data','Devices','DSSC.csv')
lat = 53.13359
lon = -1.746826

ng = Grid.load(ng)
ng = ng.add('5MWSolar', 'Solar', 0)
ng = ng.add('5MWSolarNT', 'SolarNT', 0)
ng.pvgis_fetch(device, lat, lon)
ng.match_dates()
ng.demand()
ng.carbon_emissions()


solar_capacity = 0
for asset in ng.Mix['Technologies']:
    if asset['Technology'] == 'Solar':
        solar_capacity = asset['Capacity']


farm_scaler = 5/solar_capacity

dng = Dispatch(ng)
dng = dng.set_scaler('5MWSolarNT',0)#farm_scaler * ng.DynamScale)
dng = dng.set_scaler('5MWSolar',farm_scaler/2)
dng = dng.run(1,0)

Gen = 0
for asset in dng.Distributed.Mix['Technologies']:
    if asset['Technology'] == '5MWSolar':
        Gen += asset['Generation']
    if asset['Technology'] == '5MWSolarNT':
        Gen += asset['Generation']

#print(dng.Original.CarbonEmissions[-1]/1000)
#print((dng.CarbonEmissions[-1]/1000)-(dng.Original.CarbonEmissions[-1]/1000))
Gen,Price = match_dates(Gen, Price)
solar_rev = Gen*Price

dng = Dispatch(ng)
dng = dng.set_scaler('5MWSolarNT',(farm_scaler * ng.DynamScale)/2)
dng = dng.set_scaler('5MWSolar', 0)#farm_scaler/2)#farm_scaler)
dng = dng.run(0,1)

Gen = 0
for asset in dng.Distributed.Mix['Technologies']:
    if asset['Technology'] == '5MWSolar':
        Gen += asset['Generation']
    if asset['Technology'] == '5MWSolarNT':
        Gen += asset['Generation']

Gen,Price = match_dates(Gen, Price)
solarNT_rev = Gen * Price

plt.plot(np.cumsum(solarNT_rev))


plt.xlabel('Date and Time')
plt.ylabel('Cumilative Revenue (€)')
plt.show()