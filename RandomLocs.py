import pickle

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
import random



def random_points_in_polygon(number, geodata):
    points = []
    #min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        min_x, min_y, max_x, max_y = geodata.bounds
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if geodata.contains(point):
            points.append(point)
            i += 1
    return points  # returns list of shapely point

geodata = gpd.read_file('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\Shapes\ITL1_MAY_2021_UK_BFC.shp')
geodata = geodata.to_crs(epsg=4326)
uk = [x for x in geodata['geometry']]
uk = unary_union(uk)
with open('UK.pickle', 'wb') as handle:
    pickle.dump(uk, handle, protocol=pickle.HIGHEST_PROTOCOL)

points = random_points_in_polygon(10, uk)
x = [i.x for i in points]
y = [i.y for i in points]

plt.rcParams["figure.dpi"] = 300
geodata.plot()
plt.yscale("Latitude")
plt.xlabel("Longitude")
plt.scatter(x,y, c="r")
plt.show()