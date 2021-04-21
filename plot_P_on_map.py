# -*- coding: utf-8 -*-
# 
import json
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime as dt
from  datetime import date, time, timedelta
import psycopg2 as pg2
import os.path
import json

# подключаем библиотеку подключения к монго
import mongoconnect as mc
from trainer.stantionclass import Stantion
from trainer.dbfunctions import *
from trainer.ilistweather import IListWeather

#
# многопроцессорность
#
import multiprocessing as mp


# 
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017

dt_beg = dt(2017, 5, 1)
dt_end = dt(2017, 6, 1)

# from mpl_toolkits.basemap import Basemap
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()


# координаты станций, где есть давление
lon_p        = []
lat_p        = []
val_p        = []

# станции с давлением 
stantions    = []

weather_list = get_weather_by_day(date=dt_beg,db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port)
for item in weather_list:
  item.onlyFloatOn()
  p = item.get_P()
  if p>800 and p<1300:
    lon_p.append(item.lon)
    lat_p.append(item.lat)
    val_p.append(p)
    if ( item.get_stantion() not in stantions ):
      stantions.append(item.get_stantion())

# переводим в обычный массив
val = np.array(val_p)
lon = np.array(lat_p)
lat = np.array(lon_p)

# grid data
numcols, numrows = 300, 300

mlon,mlat = (lon_p,lat_p)
mlon      = np.array(mlon)
mlat      = np.array(mlat)
xi_l      = np.linspace(min(mlon), max(mlon), numcols)
yi_l      = np.linspace(min(mlat), max(mlat), numrows)
xi, yi    = np.meshgrid(xi_l, yi_l)

lon_lat = np.c_[mlon.ravel(),mlat.ravel() ]

from scipy import interpolate
from scipy.interpolate import griddata
mmin   = (min(val)//5)*5
mmax   = max(val)
levels = np.arange(mmin,mmax,5)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()
ax.set_global()
ax.plot( lon_p,lat_p, 'o', color='red'  )

# 
# Это просто заполненные поля, но неровно
# 
zi = griddata(lon_lat, val.ravel(), (xi, yi), method='linear' )
mm = ax.contourf(xi, yi, zi, extend='both')
mc = ax.contour(xi, yi, zi,levels, colors='white',)


print(stantions, len(stantions) )


# for i, (x,y) in enumerate( zip(lon,lat), start=1 ):
  # ax.annotate(str(weather_list[i-1].get_stantion()), (x,y), xytext=(5,5),textcoords='offset points' )


plt.title("Observation data on "+str(dt))
plt.show()

exit() 