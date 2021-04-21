# -*- coding: utf-8 -*-
# 
# 
# 
# *** Пример использования ***
# 
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

from bson.son import SON

#
# многопроцессорность
#
import multiprocessing as mp

from trainer.dbfunctions import *
from statistics import mean 
import time

mongo ={'db': 'meteodb',
        # 'collection': 'meteoreport',
        'collection': 'meteo',
        'host'      : 'debian1.vka.loc',
        'port'      : 27007
        # 'port'      : 27037
        # 'host'      : 'debian1.vka.loc',
        # 'port'      : 27016
        # 'host'      : '10.10.11.120',
        # 'port'      : 27017
        }


# 
# рисуем графики
# @_times1 - pgsql
# @_times2 - mongo
# 
def plt_times(_times1,_times2,title):
  fig, (ax1,ax2) = plt.subplots(2,1)
  index = np.arange( len(date_list) )
  bar_width = 0.35
  opacity = 0.8

  ax1.bar(index, _times1, bar_width,
           alpha=opacity,
           # color='b',
           label='psql_'+title)

  ax1.bar(index + bar_width, _times2, bar_width,
            alpha=opacity,
            # color='g',
            label='mongo_'+title)

  ax1.set_title('Compare Mongodb and PGSQL Queries')
  ax1.set(xlabel='Numer of query',ylabel='Time, ms')
  # ax1.xticks(index + bar_width, date_list )
  ax1.legend()


  # 
  # second
  # 
  ax2.bar(1, [mean(_times1)], bar_width,
           alpha=opacity,
           # color='b',
           label='psql_'+title)

  ax2.bar(1 + bar_width, [mean(_times2)], bar_width,
            alpha=opacity,
            # color='g',
            label='mongo_'+title)
  ax2.set_title('Mean Queries time')
  ax2.set(xlabel='Numer of query',ylabel='Mean Time, ms')
  ax2.legend()

  plt.tight_layout()
  plt.show()

# 
# ======== PGSQL SECTION ==========
# 


# 
# тестирование нагрузки на базу постгрес
# 
def psql_surface(date,param):
  # 
  # делаем запрос в базу 
  # 
  query = "SELECT * from  get_many_surface_data_1('"+date.strftime('%Y-%m-%dT%H:%M:%SZ')+"', "+param+", NULL)"
  # считываем данные
  data = read_postgres(raw_query=query)
  print(len(list(data)))
  return data

def psql_meteogramm(date,params):
  dateend  = date-timedelta(hours=62)
  stantion = '26075'
  # 
  # делаем запрос в базу 
  # 
  query = "SELECT * from  get_many_surface_data_2( '"+stantion+"', '"+dateend.strftime('%Y-%m-%dT%H:%M:%SZ')+"','"+date.strftime('%Y-%m-%dT%H:%M:%SZ')+"' , '{ "+','.join(params)+" }' )"
  # считываем данные
  data = read_postgres(raw_query=query)
  ld = list(data)
  print( len(ld) )
  return len(ld)

def psql_aero(date,param):
  stantion = '26075'
  # 
  # делаем запрос в базу 
  # 
  query = "SELECT * from  get_one_aero_data('"+date.strftime('%Y-%m-%dT%H:%M:%SZ')+"', '"+stantion+"' )"
  # считываем данные
  data = read_postgres(raw_query=query)
  ld = list(data)
  print( len(ld) )
  return len(ld)


# 
# ======== MONGO SECTION ==========
# 

# 
# тестирование нагрузки на базу Монго
# 
def mongo_surface(date,param):
  # 
  # делаем запрос в базу 
  # 
  query = { "dt": date,"data_type" : 22, "level" : 0, "level_type" : 1,  }
 
  # считываем данные
  data   = get_query( mongo['db'], mongo['collection'], mongo['host'], mongo['port'], query )
  count    = 0
  # проходимся по ответу
  # print((data.size()))
  
  # for doc in data:
  #   count+=1
  ld = list(data)
  print( len(ld) )
  return len(ld)


# 
# тестирование нагрузки на базу Монго
# 
def mongo_aero(date,param):

  stantion = '26075'

  # 
  # делаем запрос в базу 
  # 
  query = [
            { "$match": {
              "data_type":61,
              "dt":  date,
              "station": stantion
             }
            },
            { "$project" : {
               "level":1,
               "level_type":1,
               "location":1,
               "param":1
             }
            },
            { "$unwind" : "$param" },
            { "$group": {
                "_id": {
                   "level": "$level",
                   "level_type": "$level_type",
                   "location": "$location"
                },
                "param":{ "$push": "$param" }
              }
            }
          ]
  # считываем данные
  data   = get_query_aggregate( mongo['db'], mongo['collection'], mongo['host'], mongo['port'], query )
  count    = 0

  # for doc in data:
  #   count+=1
    # print(doc)
  ld = list(data)
  print( len(ld) )
  return len(ld)

# 
# тестирование нагрузки на базу Монго
# 
def mongo_meteogramm(date,params):

  stantion = '26075'
  prevdate = date-timedelta(hours=62)
  # 
  # делаем запрос в базу 
  # 
  query = [
            {"$unwind": "$param"},
              {"$match": {
                "dt" :    { "$lte": date, "$gte": prevdate },
                "station": stantion,
                "param.descrname": { "$in" : params }
              }
              },
              {
                "$group" : {
                  "_id" : {
                    "uuid" : "$param.uuid",
                    "dt_type" : "$param.dt_type",
                    "dt_beg" : "$param.dt_beg",
                    "dt_end" : "$param.dt_end"
                  },
                  "params" : {
                  "$push" : {
                      "descrname": "$param.descrname",
                      "value" : "$param.value",
                      "code":"$param.code",
                      "quality" : "$param.quality",
                      "dt_type" : "$param.dt_type",
                      "dt_beg" : "$param.dt_beg",
                      "dt_end" : "$param.dt_end"
                    }
                  }
                }
              }, 
              {
                "$sort" : { "_id.uuid" : 1 }
              }  ]
  query = { "dt" :    { "$lte": date, "$gte": prevdate },
            "station": stantion,
            "param.descrname": { "$in" : params } }
  # считываем данные
  # data     = get_query_aggregate( mongo['db'], mongo['collection'], mongo['host'], mongo['port'], query )
  data     = get_query( mongo['db'], mongo['collection'], mongo['host'], mongo['port'], query )
  count    = 0
  # проходимся по ответу
  
  # for doc in data:
  #   count+=1
  # print(count)
  ld = list(data)
  print( len(ld) )
  return len(ld)


# numhours = 438
numhours = 64*20
# шаг по времени нужет для аэрологии
# step     = 12
step     = 64
_date    = dt(2019, 11, 25,0,0,0)
_param   = '12101' # T
# _param   = '20001' # V
# _param   = '12108' # D
# _param   = '10051' # P
# _param   = '12103' # Td
# _param   = '10061' # p
# _param   = '11001' # dd
# _param   = '11002' # ff

# список дат
# date_list = [_date - timedelta(hours=x) for x in range(0,numhours,step)]
date_list_m = [_date - timedelta(hours=x) for x in range(0,numhours,step)]
step        = 12
date_list_a = [_date - timedelta(hours=x) for x in range(0,numhours,step)]
date_list_s = [_date - timedelta(hours=x) for x in range(0,90)]



# 
# Запускаем замер времени выполнения функции
# 
def checktime(date_list,func, _param):
  _times = []
  data   = []

  for _dt in date_list:
    # 
    # замеряем время выполнения
    # 
    start_time = time.time()
    data.append(func(_dt, _param))
    # 
    # выводим время выполнения скрипта
    # 
    _t = time.time() - start_time
    _times.append(_t)

  return data, _times


# 
# Выполняем замер времени выполнения скрипта
# 

# запрос по станциям
# data1, _times1 = checktime( date_list_s, psql_surface, _param )
data2, _times2 = checktime( date_list_s, mongo_surface, _param )
# print("mean surface psql: ",mean(_times1)  )
print("mean surface mongo: ",mean(_times2) )


# запрос аэрологии
# data3, _times3 = checktime( date_list_a, psql_aero, _param )
data4, _times4 = checktime( date_list_a, mongo_aero, _param )
# print("mean aero psql: ",mean(_times3)  )
print("mean aero mongo: ",mean(_times4) )

# запрос для метеограммы за 3-е суток по разным метеопараметрам
_params = ['12101','11001','11002','12103','10051']
_params_name = ['T','Td','dd','ff','P']
# data5, _times5 = checktime( date_list_m, psql_meteogramm, _params )
data6, _times6 = checktime( date_list_m, mongo_meteogramm, _params_name )
# print("mean meteogramm psql: ",  mean(_times5) )
print("mean meteogramm mongo: ", mean(_times6) )

# 
# Сохраняем информацию в ексель
# 
# data = pd.DataFrame( {'pg_time':_times5, 'mongo_time':_times6} )
# data.to_csv('aero_time.csv',sep=";")

data = pd.DataFrame( {'mongo_deb_aero_time':_times4, 'data':data4 } )
data.to_csv('mongo2_deb_aero_time.csv',sep=";")

data = pd.DataFrame( {'mongo_deb_surf_time':_times2, 'data':data2} )
data.to_csv('mongo2_deb_surf_time.csv',sep=";")

data = pd.DataFrame( {'mongo_deb_meteogram_time':_times6, 'data':data6} )
data.to_csv('mongo2_deb_meteogram_time.csv',sep=";")

print('csv saved!')
 
# 
# Рисуем графики
# 
# plt_times(_times3,_times4,'aero')
# plt_times(_times5,_times6,'meteogramm')
