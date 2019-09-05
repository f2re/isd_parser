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

all_files      = 0.0
files_prepared = 0.0

# 
# Парсим файл и сохраняем его в базу
# 
def parse_file(csv_path):
  # создаем новый парсер
  parser = ISD()
  
  # if '\0' in open(csv_path).read():
  #   return False

  with open(csv_path, "r") as f_obj:
    reader = csv.reader( x.replace('\0', '') for x in f_obj)

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
    if parser.get_weather().get_country()!="US":
      mc.write_mongo( db=mongo_db, colletion=mongo_collection, ip=mongo_host, port=mongo_port, data=data_to_write )

    files_prepared+=1
    if files_prepared>0:
      print("Operation worked "+str(all_files)+"/"+str(files_prepared)+" "+str(100*files_prepared/all_files)+"%" )
  return "ok"

# 
# Настройки многопоточности
# 
pool = Pool(processes=12)
logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)

paths = ["/home/ivan/WEATHERSRC/2016"]
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
results   = pool.map(parse_file,file_list) 
pool.close()
pool.join()
print(results)
