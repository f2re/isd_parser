# -*- coding: utf-8 -*-
# 
# 

import json
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime as dt
from  datetime import date, time, timedelta

import os.path

# многопроцессорность
#
import multiprocessing as mp

import numpy as np

from trainer.dbfunctions import *

# 
# параметры подключения к монго
# 
mongo               = {}
mongo['db']         = 'srcdata'
mongo['collection'] = 'meteoisd'
mongo['host']       = 'localhost'
mongo['port']       = 27017

# 
# начало периода
# для данных, на которых обучаемся
# 
dt_cur  = dt(1994, 1, 1)
n_timesteps = 9850


stantion                  = '260630'
# stantion                  = '227680'
# stantion                  = '227490'

params                    = ['T']


_weather_list, \
_from_date,    \
_to_date = get_weather_on_ISDstation(date       = dt_cur,
                                st         = stantion,
                                offset     = 0,
                                step       = n_timesteps,
                                db         = mongo['db'], 
                                collection = mongo['collection'], 
                                ip         = mongo['host'], 
                                port       = mongo['port'])

data = pd.DataFrame(columns=['Температура, С','Точка росы','Давление, Гпа','Скорость ветра','Направление ветра','Видимость','Явления','Влажность'])
for item in _weather_list.get_all():
  data.loc[item.get_date()] = [item.get_T(),item.get_Td(),item.get_P(),item.get_ff(),item.get_dd(),item.get_VV(),item.get_WW(),item.get_RR()]
  print(item.get_date())

data.to_csv(stantion+".csv")
exit() 