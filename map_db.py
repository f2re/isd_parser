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
# диапазон дат 
# 
def daterange(start_date, end_date):
  for n in range(int ((end_date - start_date).days)):
    yield start_date + timedelta(n)


# 
# Создаем массив с фвлениями
# 
def create_yavl_dict(daytime):

  weather = IListWeather( get_weather_by_day( daytime ) )

  # 
  # запбираем список станций в погоде
  # 
  stations = weather.get_allst()

  # 
  # проходимся по станциям и выясняем какие явления там есть
  # 
  for st in stations:
    # 
    # получаем погоду за день по одной станции
    # 
    filtered = weather.get_filter_by_stantion(st)

    # 
    # а теперь получем продолжительность явлений 
    # 
    longs = filtered.get_yavl_longs()

    for l,v in longs.iteritems():
      if l in yavl:
        yavl.update( {l: yavl[l]+v } )
      else:
        yavl.update( {l: v } )

  return yavl

# 
# 
# =======================================================
# 
# 

# 
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017

dt = dt(2017, 5, 1)

# from mpl_toolkits.basemap import Basemap
import cartopy
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()

# m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=80,\
            # llcrnrlon=0,urcrnrlon=200,lat_ts=20,resolution='l',ax=ax)

st_list = [

  '106190', # IDAR-OBERSTEIN
  '107380', # STUTTGART/SCHNARRENBERG
  '115180', # PRAHA-LIBUS
  '119340', # POPRAD-GANOVCE
  '132720', # BEOGRAD/KOSUTNJAK
  '260380', # TALLINN-HARKU
  '484315', # NAKHON PHANOM
  '485650', # PHUKET AIRPORT
  '034185', # NOTTINGHAM
  '038090', # CAMBORNE
  '071460', # TRAPPES
  '076460', # NIMES-COURBESSAC
  '170310', # SAMSUN
  '170630', # ISTANBUL BOLGE (KARTAL)
  '171310', # ANKARA/CENTRAL
  '234180', # PECHORA Печора
  '238040', # SYKTYVKAR Сыктывкар
  '249590', # JAKUTSK Якутск
  '267810', # SMOLENSK SMOLENSK
  '277300', # RYAZAN
  '284400', # EKATERINBURG (VERHNEE DUBROVO) KOLTSOVO
  '287220', # UFA-DIOMA 
  '319741', # VLADIVOSTOK (SAD GOROD) KNEVICHI
  '287854', # 287854 ABAKAN  53.74 91.385  RSM00029862
  '353942', # 353942 ASTRAKHAN 46.283333 48.006278 RSM00034882
  '377352', # 377352 UYTASH  42.816822 47.652294 RSM00037259
  '347300', # 347300 ROSTOV NA DONU  47.258208 39.818089 RSM00034731
  '345600', # 345600 GUMRAK  48.782528 44.345544 RSM00034467
  '313289', # 313289 IGNATYEVO 50.425394 127.412478  RSM00031510
  '341720', # 341720 TSENTRALNY  51.565  46.046667 RSM00034172
  '351210', # 351210 ORENBURG  51.795786 55.456744 RSM00035121
  '341220', # 341220 CHERTOVITSKOYE  51.814211 39.229589 RSM00034122
  '307100', # 307100 IRKUTSK 52.268028 104.388975  RSM00030715
  '296340', # 296340 TOLMACHEVO  55.012622 82.650656 RSM00029634
  '303090', # 303090 BRATSK  56.370556 101.698331  RSM00030309
  '282250', # 282250 BOLSHOYE SAVINO 57.914517 56.021214 RSM00028225
  '259130', # 259130 SOKOL 59.910989 150.720439  RSM00025913
  '234710', # 234710 NIZHNEVARTOVSK  60.949272 76.483617 RSM00023955
  '218230', # 218230 YAKUTSK 62.09325  129.770672  RSM00024959
  '225500', # 225500 TALAGI  64.600281 40.716667 RSM00022543
  '221130', # 221130 MURMANSK  68.781672 32.750822 RSM00022217
  # '261148', # Минск
  # '225500', # TALAGI Архангельск

  # '260630', # Левашово
  # '262580', # Псков
  # '370540', # Мин воды
  # '370010', #витязево
  # '339460', #SIMFEROPOL
  # '345600', #GUMRAK
  # '274590', #STRIGINO
  # '221130', #MURMANSK
  # '286420', #BALANDINO
  # '234710', #NIZHNEVARTOVSK
  # '296420', #KEMEROVO
  # '287854', #ABAKAN
  # '307100', #ИРКУТСК
  # '313289', #IGNATYEVO
  # '319741', #KNEVICHI
  
  # '401800', #BEN GURION
  # '173725', #HATAY
]


lon = []
lat = []

# координаты станций, где есть давление
lon_p = []
lat_p = []

# координаты станций, где которые есть в списке
lon_st = []
lat_st = []

weather_list2 = []


# 
# Наносим станции зондирования
# 
# df = pd.read_fwf('TRAIN/igra2-station-list.txt', 
#                   widths=[ 12,9,10,10,31,4,5,7 ],
#                   names=[ 'st','lat','lon','height','name','dt_from','dt_to','count' ])

# df = df[ ( (df['dt_to']==2020) & (df['dt_from']<=2000)) ]
# # print(df)

# for index,row in df.iterrows():
#   lat.append(row.lat)
#   lon.append(row.lon)

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.stock_img()
# # ax.plot( lon,lat, 'o'  )
# ax.coastlines()
# ax.set_global()
# ax.plot( lon,lat, 'o', color='red'  )

# for i, (x,y) in enumerate( zip(lon,lat), start=1 ):
#   # print(weather_list[i-1])
#   ax.annotate( str(df.iloc[i-1]['name']), (x,y), xytext=(3,3), textcoords='offset points' )




# weather_list = get_weather_by_ISDdayNonStandartSrok(date=dt,db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port)
weather_list = get_weather_by_day(date=dt,db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port)

arr = []

for item in weather_list:
  lat.append(item.lat)
  lon.append(item.lon)
  # if item.get_P()!='' and item.get_country()=="RS":
  if item.get_country()=="RS":
    lon_p.append(item.lon)
    lat_p.append(item.lat)
    arr.append([ item.get_stantion(), item.get_name(), item.get_lat(), item.get_lon() ])
    # print(item)
  if item.get_stantion() in st_list:
    # weather_list2.append(item)
    weather_list2.append(item)
    # print(item.get_stantion())
    lon_st.append(item.lon)
    lat_st.append(item.lat)

# df = pandas.DataFrame(columns=['st','name','lat','lon'])
# df = pd.DataFrame(arr)
# df.to_csv('airports.csv', sep=";")
# exit()

# X,Y = m(lon,lat)


ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
# ax.plot( lon,lat, 'o'  )
ax.coastlines()
ax.set_global()


# m.drawcoastlines()
# m.bluemarble(scale=0.5)
# m.fillcontinents(color='white',lake_color='lightblue')
# draw parallels and meridians.
# m.drawparallels(np.arange(-90.,91.,10.))
# m.drawmeridians(np.arange(-180.,181.,10.))
# m.drawmapboundary(fill_color='lightblue')


ax.plot( lon_p,lat_p, 'o', color='red'  )
# ax.plot( lon_st,lat_st, 'o', color='red'  )
# ax.scatter(lon_p,lat_p)


# for i, (x,y) in enumerate( zip(lon,lat), start=1 ):
#   ax.annotate(str(weather_list[i-1].get_stantion()), (x,y), xytext=(5,5),textcoords='offset points' )

# for i, (x,y) in enumerate( zip(lon_st,lat_st), start=1 ):
for i, (x,y) in enumerate( zip(lon_p,lat_p), start=1 ):
  # print(weather_list[i-1])
  # ax.annotate( str(weather_list[i-1].get_name() ), (x,y), xytext=(5,-5), textcoords='offset points' )
  ax.annotate( str(arr[i-1][1])+" "+str(arr[i-1][0]), (x,y), xytext=(-10,-10), textcoords='offset points' )

plt.title("Observation data on "+str(dt))
plt.show()

exit() 