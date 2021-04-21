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

from trainer.stantionclass import Stantion

from trainer.ilistweather import IListWeather

#
# многопроцессорность
#
import multiprocessing as mp

from trainer.LSTMTrainerClass import LSTMTrainerClass
from trainer.dbfunctions import get_weather_on_ISDstation

# 
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017

# 
# начало периода
# для данных, на которых обучаемся
# 
dt_cur  = dt(2009, 1, 1)
# 
# начало периода для данных, на которых проверяем
# 
dt_pred = dt(2017, 1, 1)


import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.backend as K

num_epochs                = 600
# входные параметры
# количество входных параметров
n_inputs                  = 11
# 
# празмер батча
# 
batch_size                = 32
# шаг по времени (дни)
n_timesteps               = 2830
# шаг по времени для прогнозов
n_timesteps_pred          = 1050
# количество часов назад
# еще это может быть length
n_backtime_delta          = 3
# прогноз на который час
n_forecastdelta           = 24
# какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
n_dropout                 = 0.2
# станция, которую ищем, по которой будем обучаться
# в этой базе в конце номера станций надо добавлять ноль (0)
stantion                  = '340560'
# параметры, по которым прогнозируем
params                    = ['T']
# params                    = ['T','P','dd','ff']


trainer = LSTMTrainerClass()

# 
# Настриваем тренировочную базу
# 
trainer.set_stantion( stantion )   \
       .set_dt( dt_cur )           \
       .set_dt_predicted( dt_pred )   \
       .set_params(params)             \
       .set_timesteps( n_timesteps ) \
       .set_timestep_predicted( n_timesteps_pred )   \
       .set_backtimedelta( n_backtime_delta ) \
       .set_forecastdelta( n_forecastdelta ) \
       .set_inputs(n_inputs)   \
       .set_batch_size(batch_size)   \
       .set_plot(True)         \
       .set_model_name("repairStantion") \
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