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


# импортируем класс станции
from stantionclass import Stantion
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
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017


# 
# Парсим файл и сохраняем его в базу
# 
def parse_file(csv_path):
  # csv_path = "srcdata/72405503714.csv"
  # csv_path = "srcdata/99999953155.csv"
  # csv_path = "srcdata/03023099999.csv"
  # csv_path = "srcdata/A5125600451.csv"
  # создаем новый парсер
  parser = ISD()
  
  with open(csv_path, "r") as f_obj:

    reader = csv.reader(f_obj)

    # счетчик линий
    linenum = 0

    data_to_write=[]

    for row in reader:
      if linenum==0:
        # задаем заголовок
        parser.set_header( row )
      else:
        # задаем коды
        parser.set_code( row ) \
              .parse()
        data_to_write.append( parser.get_weather().to_mongo() )
        # print parser.get_weather()
      linenum+=1
    
    # записываем это все в базу
    mc.write_mongo( db=mongo_db, colletion=mongo_collection, ip=mongo_host, port=mongo_port, data=data_to_write )

    # 
    # Сохраняем название последнего обработанного файла
    # 
    # lastfile={'file':csv_path}
    # with open(lastfilefile, 'w') as outfile:
    #   json.dump(lastfile, outfile)

  return "ok"

# 
# последний файл, который мы обработали
# 
lastfilefile = 'lastfile.txt'

# 
# Настройки многопоточности
# 
pool = Pool(processes=12)
logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)




# создаем файл с настройками (последним обработанным файлом)
lastfile={'file':''}
if ( os.path.exists(lastfilefile) ):
  with open(lastfilefile) as json_file:  
    data = json.load(json_file)


path = "/home/ivan/WEATHERSRC/2018"
file_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
  for file in f:
    if '.csv' in file:
      file_list.append(os.path.join(r, file))

print file_list
# 
# запуливаем все в треды
# 
results = []
results = pool.map(parse_file,file_list) 
pool.close()
pool.join()


