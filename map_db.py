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

import pprint

from collections import defaultdict
from collections import Counter

# подключаем библиотеку подключения к монго
import mongoconnect as mc

from stantionclass import Stantion

from ilistweather import IListWeather

#
# многопроцессорность
#

import multiprocessing as mp


# 
# 
#  Считываем погоду по дням
#  

def get_weather_by_day( date, db, collection, ip, port ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)

  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=0,minute=0,second=59 )
  # ищем станции рядом 
  query = { "date": { "$gte":from_date , "$lte":to_date }, "station.country":"RS"  }
  
  res      = db.find(query)
  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    weather.append( st.fromMongoISDFormat(doc) )
  print count
  # возвращаем станции
  return weather

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

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()

m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=200,lat_ts=20,resolution='l',ax=ax)

lon = []
lat = []
weather_list = get_weather_by_day(date=dt,db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port)
for item in weather_list:
  lat.append(item.lat)
  lon.append(item.lon)
X,Y = m(lon,lat)

m.drawcoastlines()
m.bluemarble(scale=0.5)
# m.fillcontinents(color='white',lake_color='lightblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,10.))
m.drawmeridians(np.arange(-180.,181.,10.))
m.drawmapboundary(fill_color='lightblue')




ax.scatter(X,Y)

# for i, (x,y) in enumerate( zip(X,Y), start=1 ):
#   ax.annotate(str(weather_list[i-1].get_stantion()), (x,y), xytext=(5,5),textcoords='offset points' )

plt.title("Observation data on "+str(dt))
plt.show()

exit() 