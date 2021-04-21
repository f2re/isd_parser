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
from trainer.ForecastLSTMTrainerClass import ForecastLSTMTrainerClass
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
dt_cur  = dt(2003, 1, 1)
# dt_cur  = dt(2017, 1, 1)
# 
# начало периода для данных, на которых проверяем
# 
dt_pred = dt(2016, 7, 21)


# 
# количество эпох
# 
num_epochs                = 200
# 
# празмер батча
# 
batch_size                = 32

# входные параметры
# количество входных параметров
n_inputs                  = 10
# шаг по времени (дни)
n_timesteps               = 4950
# n_timesteps               = 550
# шаг по времени для прогнозов
n_timesteps_pred          = 1200
# n_timesteps_pred          = 200

# количество измерений назад
# еще это может быть length
n_backtime                = 12
# 
# а это шаг по времени, когда мы отсутпаем вперед
# чтобы найти есть ли через это время явление
# например, 3 часа
# 
n_backtime_delta          = 3
# какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
n_dropout                 = 0.2
# станция, которую ищем, для которой будем восстанавливать значения
# в этой базе в конце номера станций надо добавлять ноль (0)
# stantion                  = '370540' # Мин воды
# stantion                  = '261148' # Минск
# stantion                  = '260630' # Левашово
# stantion                  = '262580' # Псков
# параметры, по которым прогнозируем
params                    = ['T']


st_list = [
  # '370010', #витязево
  # '339460', #SIMFEROPOL
  # '345600', #GUMRAK
  # '274590', #STRIGINO
  # '225500', #TALAGI
  # '221130', #MURMANSK
  # '286420', #BALANDINO
  # '234710', #NIZHNEVARTOVSK
  # '296420', #KEMEROVO
  # '287854', #ABAKAN
  # '307100', #ИРКУТСК
  # '313289', #IGNATYEVO
  # '319741', #KNEVICHI
  
  # '401800', #BEN GURION
  # '173725', #HATAY
]

# делаем цикл по станциям
for stantion in st_list:
  # 
  # создаем класс
  # 
  trainer = ForecastLSTMTrainerClass()

  # 
  # Настриваем тренировочную базу
  # 
  trainer.set_stantion( stantion )   \
         .try_fill_stantion_name() \
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
         .set_load(True)        \
         .set_shuffle(True)   \
         .set_processes(9)      \
         .set_dbtype( 'isd' )        \
         .set_epochs( num_epochs )    \
         .set_model_name("OYAP_CONV")    

  # 
  # тренируем или используем загруженные модели
  # 

  trainer.load()    \
         .train()
  # end loop
  
exit() 

# 
# загружаем данные из файла и обучаем на загруженных данных
# 
# trainer.fillfromdt(dt_cur,n_timesteps) \
#        .train()
# exit()


# 
# Прогнозируем по сораненным данным
# загружаем модель
# 
# 
trainer.load_model(params[0])
trainer.fillfromdt_predict(dt_pred,n_timesteps_pred) \
       .load_predicted() \
       .mkbatch_predict( params[0] ) \
       .split_batch()      \
       .scale()            \
       .plot_predict(show=True,param=params[0],save=True)
exit()
# trainer.predict(params[0])

# 
# Здесь загружаем модель из файла
# задаем входные параметры и прогнозируем 
# 

# загружаем модель
trainer.load_model(params[0])

data      = [[
# 
#               T Td    P    dd   V    ff    hour  surise  sunset day month
# 
              [11,  10,  1013.9,  0, 11265, 0, 19.00, 1.60,  18.71, 21,  6], # 1  #
              [11,  10,  1013.9,  0, 11265, 0, 19.30, 1.60,  18.71, 21,  6], # 2  #
              [10,  10,  1013.9,  0, 6000,  0, 19.50, 1.60,  18.71, 21,  6], # 3  #
              [10,  9, 1013.9,  0, 9000,  0, 20.00, 1.60,  18.71, 21,  6], # 4  #
              [9, 9, 1013.9,  0, 9000,  0, 20.50, 1.60,  18.71, 21,  6], # 5  #
              [9, 9, 1013.9,  0, 11265, 0, 21.00, 1.60,  18.71, 21,  6], # 6  #
              [10,  10,  1013.9,  240, 11265, 1, 21.50, 1.60,  18.71, 21,  6], # 7  #
              [11,  10,  1013.9,  240, 11265, 1, 22.00, 1.60,  18.71, 21,  6], # 8  #
              [11,  10,  1013.9,  240, 11265, 1, 22.15, 1.60,  18.71, 21,  6], # 9  #
              [10,  10,  1013.9,  0, 9900,  0, 22.50, 1.60,  18.71, 21,  6], # 10 #
              [10,  10,  1013.9,  220, 9900,  1, 23.00, 1.60,  18.71, 21,  6], # 11 #
              [10,  10,  1014.9,  240, 9900,  1, 23.50, 1.60,  18.71, 21,  6], # 12 #
]]

import numpy as np

shp = np.shape(data)

X_train  = np.reshape( 
                      trainer._scalerX.transform(np.reshape(data, ( shp[0]*shp[1], shp[2]) ))
                      , (shp[0], shp[1], shp[2])
                    )
# прогнозируем значения
predicted = trainer.custom_predict( X_train )

result = predicted[0][0]
print( "See on OYAP: ", result )
    
if result[0] == 1.0 and result[1] == 0.0:
  print('OYAP esist!')
elif result[0] == 0.0 and result[1] == 1.0:
  print('NO OYAP')
else:
  print("undefined OYAP")

oyap_beg = predicted[1][0]
oyap_val = predicted[2][0]

from math import modf

print( oyap_beg, "Туман сядет через ", int(modf(oyap_beg)[1]) ,  "часов ", int(modf(oyap_beg)[0]*60), "минут"  )
print( oyap_val, "продлится ", modf(oyap_val)[1] , " часов ", int(modf(oyap_val)[0]*60), "минут"  )

print( predicted )

# трансформируем


exit() 