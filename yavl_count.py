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
# считываем json файл и раскодируем его
# 
def parse_json(filename = ''):
  if filename!='':

    # массив данных по строкам
    # datas=[]

    for line in open(filename,'r'):
      # загружаем строку в джсон
      js = json.loads(line)

      if js['city']['country']=="RU" :
        print json.dumps(js,indent=4)
        return

  else:
    return False
  return 



# 
# 
#  Считываем погоду по дням
#  

def get_weather_by_day( date ):
  # подключаемся к БД
  db = mc.get_mongodb(db='meteodb', collection="meteoreport", ip='10.10.11.120', port=27017,)

  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=23,minute=59,second=59 )
  # print from_date
  # print to_date
  # ищем станции рядом 
  query = { "station_type": 2,
            "dt": { "$gte":from_date , "$lte":to_date },
            "level":0,
            "level_type":1,
            "data_type" : 48, # kAerodrome = 48,
          }
  print query
  res      = db.find(query)
  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    weather.append( st.fromMongoFormat(doc) )
  # возвращаем станции
  return weather

# 
# диапазон дат 
# 
def daterange(start_date, end_date):
  for n in range(int ((end_date - start_date).days)):
    yield start_date + timedelta(n)

# 
# получаем погоду на станции 
# @st   - станция
# @offset - смещение в днях от начала
# @step - шаг в днях
# 
def get_weather_on_station(st, offset=0, step=100):
  # подключаемся к БД
  db = mc.get_mongodb(db='srcdata', collection="meteoreport", ip='localhost', port=27017,)

  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = dt(1995, 1, 1, 1) + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )

  # ищем станции рядом 
  query = { "date": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "stantion_id":st }

  print query
  # делаем запрос в базу
  res     = db.find(query)
  weather = []
  count   = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    weather.append( doc )

  print count
  # возвращаем станции
  return weather



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
#  =======================

# 
# получаем станции в радиусе одного градуса рядом с координатами
# 

from multiprocessing import Process, Pool

i=0

yavl = {}

# запускаем цикл в котором будем считывать данные
start_date = dt(2018, 7, 1)
end_date = dt(2019, 6, 30)

pool = Pool(processes=12)

dates = []
for daytime in daterange(start_date, end_date):
  dates.append(daytime)
# 
# запуливаем все в треды
# 
results = []
results = pool.map(create_yavl_dict,dates)
pool.close()
pool.join()

for r in results:
  for l,v in r.iteritems():
    if l in yavl:
      yavl.update( {l: yavl[l]+v } )
    else:
      yavl.update( {l: v } )

print yavl

i=0
inrow = len(yavl)/2+1
fig, axs = plt.subplots(2, inrow, sharey=True,sharex=True, tight_layout=True)

row = 0
# 
# проходимся по явлениями
# 
for key,num in yavl.iteritems():
  if i>=inrow:
    row=row+1
    i=0
  # 
  # считаем повторяемость каждого явления
  # 
  # print num
  # print Counter(num).items()
  axs[row][i].hist( num )
  axs[row][i].set_title(key)
  print key
  i+=1

plt.savefig("yavl_long.eps", format='eps', dpi=1200)
plt.show()