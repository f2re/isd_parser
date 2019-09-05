# -*- coding: utf-8 -*-
# 
import json
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime as dt
from  datetime import date, time, timedelta

import os.path

from collections import defaultdict
from collections import Counter

from stantionclass import Stantion

from ilistweather import IListWeather

#
# многопроцессорность
#

import multiprocessing as mp

from trainer.LSTMTrainerClass import LSTMTrainerClass
from trainer.dbfunctions import get_weather_on_ISDstation


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

# начало периода
dt_cur = dt(2017, 1, 1)


import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.backend as K

num_epochs                = 2
# входные параметры
# количество входных параметров
n_inputs                  = 11
# шаг по времени (дни)
n_timesteps               = 700
# количество измерений назад
# еще это может быть length
n_backtime                = 3
# какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
n_dropout                 = 0.2


trainer = LSTMTrainerClass()

# 
# Настриваем тренировочную базу
# 
trainer.set_stantion( '340560' )   \
       .set_dt( dt_cur )           \
       .set_params(['T','P','dd','ff'])             \
       .set_timesteps( n_timesteps ) \
       .set_inputs(n_inputs)   \
       .set_plot(True)         \
       .set_save(True)         \
       .set_load(False)        \
       .set_dbtype( 'isd' )        \
       .set_epochs( num_epochs )        

# 
# Запускаем процесс
# 
trainer.load() \
       .train()


exit() 