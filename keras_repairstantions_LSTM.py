# -*- coding: utf-8 -*-
# 
# 
# 
# *** Пример использования ***
# 
# 
# 
import json
from  datetime import datetime as dt
from  datetime import date, time, timedelta

from trainer.LSTMTrainerClass import LSTMTrainerClass
from trainer.RepairLSTMTrainerClass import RepairLSTMTrainerClass
from trainer.Laplacian import Laplacian


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
dt_cur  = dt(2007, 1, 1)
# 
# начало периода для данных, на которых проверяем
# 
dt_pred = dt(2018, 1, 1)


# 
# количество эпох
# 
num_epochs                = 50
# 
# празмер батча
# 
batch_size                = 32

# входные параметры
# количество входных параметров
n_inputs                  = 30
# шаг по времени (дни)
n_timesteps               = 3300
# шаг по времени для прогнозов
n_timesteps_pred          = 530

# количество измерений назад
# еще это может быть length
n_backtime                = 3
# 
# а это шаг по времени, когда мы отсутпаем назад
# например, 3 часа
# 
n_backtime_delta          = 3
# какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
n_dropout                 = 0.2
# станция, которую ищем, для которой будем восстанавливать значения
# в этой базе в конце номера станций надо добавлять ноль (0)
# stantion                  = '275185'
stantion                  = '123750'
# stantion                  = '075350'
# stantion                  = '072660'
# 
# станции, на которых учимся (у которых есть зависимости)
# 
custom_stantions          = ['267810','268820','262580','262420','261350','269970','265850']
# параметры, по которым прогнозируем
params                    = ['T']

# 
# создаем класс
# 
trainer = RepairLSTMTrainerClass()

# 
# Настриваем тренировочную базу
# 
trainer.set_stantion( stantion )   \
       .set_stantions( custom_stantions ) \
       .set_dt( dt_cur )           \
       .set_dt_predicted( dt_pred )   \
       .set_timestep_predicted( n_timesteps_pred )   \
       .set_params(params)             \
       .set_timesteps( n_timesteps ) \
       .set_backtimedelta( n_backtime_delta ) \
       .set_backtime( n_backtime ) \
       .set_inputs(n_inputs)   \
       .set_batch_size(batch_size)   \
       .set_plot(True)         \
       .set_save(True)         \
       .set_load(False)        \
       .set_shuffle(False)   \
       .set_processes(10)      \
       .set_dbtype( 'isd' )        \
       .set_epochs( num_epochs )    \
       .set_model_name("repair_LSTM")    

trainer.fill_stantions() \
       .plot_stantions() \
       .plot_corrmatrix('T') \
       .load()           \
       .train()         
       # .set_onlyvalues() \
       # 
       # 
       # .plot_corrmatrix('dd') \
       # .plot_corrmatrix('ff') \
       # .plot_corrmatrix('P') \
       # find_stantion() \
       # .plot_stantions() 
       # .load()          \
       # .train()         

# 
# Запускаем процесс
# 
# trainer.load() \
       # .train()


exit() 