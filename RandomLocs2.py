import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
import random



def random_points_in_polygon(number, geodata,geodata2):
    points = np.zeros(number+1,dtype=Point)
    #min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    min_x, min_y, max_x, max_y = geodata.bounds
    while i <= number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        for j in range(len(geodata2)):
            if geodata2.iloc[j].contains(point):
                points[i] = point
                i += 1
                continue
    return points  # returns list of shapely point

geodata = gpd.read_file('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Shapes\PCON_DEC_2020_UK_BFC.shp')
geodata = geodata.to_crs(epsg=4326)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Na h-Eileanan an Iar'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Orkney and Shetland'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Foyle'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'West Tyrone'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Fermanagh and South Tyrone'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Newry and Armagh'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'South Down'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Newry and Armagh'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Mid Ulster'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'East Londonderry'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'North Antrim'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'South Antrim'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Upper Bann'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Lagan Valley'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Strangford'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'East Antrim'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'North Down'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Belfast East'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Belfast South'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Belfast West'], inplace=True)
geodata.drop(geodata.index[geodata['PCON20NM'] == 'Belfast North'], inplace=True)


with open('UK.pickle', 'rb') as handle:
    uk = pickle.load(handle)
    
points = random_points_in_polygon(1000, uk, geodata['geometry'])
x = [i.x for i in points]
y = [i.y for i in points]
D = pd.DataFrame()
D['Latitude'] = y
D['Longitude'] = x
D.to_csv('RandomLocs50.csv')
plt.rcParams["figure.dpi"] = 300
geodata.plot()
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.scatter(x,y, c="r",s=2)
plt.show()