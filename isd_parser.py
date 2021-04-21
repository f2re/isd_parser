# -*- coding: utf-8 -*-
import csv
from datetime import datetime, timedelta

# то, что нужно для монги
import os.path
import os
import json
from pymongo import MongoClient
import numpy as np
import pandas as pd
# подключаем библиотеку подключения к монго
import mongoconnect as mc
import trainer.cassandraconnect as cc


# импортируем класс станции
from trainer.stantionclass import Stantion
# списки погоды
from ilistweather import IListWeather
# класс работы со строками
from isd import ISD

#
# многопроцессорность
#
import multiprocessing, logging
from multiprocessing import Process, Pool

# 
# mongo or cassandra splitter
# 
# m_or_c = "mongo"
m_or_c = "cassandra"

# 
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017

cassandra_host     = 'localhost'
cassandra_port     = 9042

all_files      = 0.0
files_prepared = 0.0

# 
# Парсим файл и сохраняем его в базу
# 
def parse_file(csv_path):
  # создаем новый парсер
  parser = ISD()

  # connect to cassandra
  # and get client
  client = cc.get_connector(ip=cassandra_host, port=cassandra_port)
  
  # 
  # сохраняем станцию отдельно
  # 
  stantion = None

  with open(csv_path, "r") as f_obj:
    reader = csv.reader( x.replace('\0', '') for x in f_obj)

    # счетчик линий
    linenum = 0

    data_to_write = []
    latlon_id     = None
    st_id         = None

    # 
    # сохраняем станцию
    # 
    # запрос в базу на запись

    for row in reader:
      if linenum==0:
        # задаем заголовок
        parser.set_header( row )
      else:
        # задаем коды
        parser.set_code( row ) \
              .parse()
        _weather  = parser.get_weather()
        _weather.onlyFloatOn()
        # 
        # Stantions
        # 
        if stantion is None:
          if ( m_or_c=="mongo" ):
            stantion = parser.get_weather().to_stantion_mongo()
          else:
            _ll, _st  = _weather.to_stantion_cassandra()
            stantion  = True

            # 
            # latlon
            # 
            _s = "SELECT * FROM stantionskeyspace.locations WHERE lat=? AND lon=? LIMIT 1 ALLOW FILTERING"
            latlon = cc.search( _s, data=[_weather.get_lat(),_weather.get_lon()], client=client )
            one = latlon.one()
            if one is not None:
              latlon_id = one.locationid
            else:
              latlon_id = _ll[1][0]
              cc.query_all( [_ll], client=client )

            # 
            # Stantions
            # 
            _s = "SELECT * FROM stantionskeyspace.stantions WHERE number=? LIMIT 1 ALLOW FILTERING"
            st = cc.search( _s, data=[_weather.get_stantion()], client=client )
            one = st.one()
            if one is not None:
              st_id = one.stantionid
            else:
              st_id = _st[1][0]
              # добавляем айди координат
              _st[1].append(latlon_id)
              cc.query_all( [_st], client=client )


        # 
        # data to write preparing
        # 
        _data_to_write = False
        if ( m_or_c=="mongo" ):
          _data_to_write = parser.get_weather().to_mongo()
        else:
          # print(st_id, latlon_id)
          _data_to_write = parser.get_weather().to_cassandra( st_id, latlon_id )

        # append to batch
        data_to_write.append( _data_to_write )
        # print parser.get_weather()
      linenum+=1
    
    # записываем это все в базу
    if parser.get_weather().get_country()!="US" and parser.get_weather().get_country()!="CA":
      if ( m_or_c=="mongo" ):
        mc.write_mongo( db=mongo_db, colletion=mongo_collection, ip=mongo_host, port=mongo_port, data=data_to_write )
      else:
        # prepare query
        print("query begin...")
        for d in data_to_write:
          cc.query_all(d,client=client)
        print("query end!")

  # 
  # записываем данные по станции
  # 
  if stantion is not None:
    if ( m_or_c=="mongo" ):
      mc.write_mongo( db=mongo_db, colletion='isd_stantions', ip=mongo_host, port=mongo_port, data=[stantion] )

  return "ok"

# 
# Настройки многопоточности
# 
pool = Pool(processes=14)
logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)

paths = [
         "/home/ivan/WEATHERSRC/2005",
         # "/home/ivan/WEATHERSRC/2006",
         # "/home/ivan/WEATHERSRC/2007",
         # "/home/ivan/WEATHERSRC/2008",
         # "/home/ivan/WEATHERSRC/2009",
         # "/home/ivan/WEATHERSRC/2010",
         # "/home/ivan/WEATHERSRC/2011",
         # "/home/ivan/WEATHERSRC/2012",
         # "/home/ivan/WEATHERSRC/2013",
         # "/home/ivan/WEATHERSRC/2014",
         # "/home/ivan/WEATHERSRC/2015",
         # "/home/ivan/WEATHERSRC/2016",
         # "/home/ivan/WEATHERSRC/2017",
         # "/home/ivan/WEATHERSRC/2018",
         # "/home/ivan/WEATHERSRC/2019",
         ]
file_list = []
for path in paths:
  # path = "/home/ivan/WEATHERSRC/2018"
  # r=root, d=directories, f = files
  for r, d, f in os.walk(path):
    for file in f:
      if '.csv' in file:
        file_list.append(os.path.join(r, file))

# print file_list
# 
# запуливаем все в треды
# 
results   = []
all_files = len(file_list)

# results   = parse_file(file_list[0]) 

results   = pool.map(parse_file,file_list) 
pool.close()
pool.join()
print(results)
