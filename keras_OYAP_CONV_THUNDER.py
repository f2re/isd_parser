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

from trainer.ForecastCONVAeroThunder import ForecastCONVAeroThunder

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
dt_cur  = dt(2005, 1, 1)
# dt_cur  = dt(2017, 1, 1)
# 
# начало периода для данных, на которых проверяем
# 
dt_pred = dt(2016, 7, 21)


# 
# количество эпох
# 
num_epochs                = 50
# празмер батча
# 
batch_size                = 32

# входные параметры
# количество входных параметров
n_inputs                  = 25
# шаг по времени (дни)
n_timesteps               = 3650
# n_timesteps               = 550
# шаг по времени для прогнозов
n_timesteps_pred          = 1000
# n_timesteps_pred          = 300

# количество измерений назад
# еще это может быть length
n_backtime                = 12
# 
# а это шаг по времени, когда мы отсутпаем вперед
# чтобы найти есть ли через это время явление
# например, 3 часа
# 
n_backtime_delta          = 12
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

  # '225500', # TALAGI Архангельск
  # ['234180',"RSM00023415-data.txt"], # PECHORA Печора
  # ['238040',"RSM00023802-data.txt"], # SYKTYVKAR Сыктывкар
  # ['249590',"RSM00024959-data.txt"], # JAKUTSK Якутск
  ['267810',"RSM00026781-data.txt"], # SMOLENSK SMOLENSK
  ['277300',"RSM00027730-data.txt"], # RYAZAN
  ['284400',"RSM00028445-data.txt"], # EKATERINBURG (VERHNEE DUBROVO) KOLTSOVO
  ['287220',"RSM00028722-data.txt"], # UFA-DIOMA 
  # ['319741',"RSM00031977-data.txt"], # VLADIVOSTOK (SAD GOROD) KNEVICHI
  ['287854',"RSM00029862-data.txt"], # 287854 ABAKAN  53.74 91.385  RSM00029862
  # ['353942',"RSM00034882-data.txt"], # 353942 ASTRAKHAN 46.283333 48.006278 RSM00034882
  # ['377352',"RSM00037259-data.txt"], # 377352 UYTASH  42.816822 47.652294 RSM00037259
  ['347300',"RSM00034731-data.txt"], # 347300 ROSTOV NA DONU  47.258208 39.818089 RSM00034731
  ['345600',"RSM00034467-data.txt"], # 345600 GUMRAK  48.782528 44.345544 RSM00034467
  # ['313289',"RSM00031510-data.txt"], # 313289 IGNATYEVO 50.425394 127.412478  RSM00031510
  # ['341720',"RSM00034172-data.txt"], # 341720 TSENTRALNY  51.565  46.046667 RSM00034172
  # ['351210',"RSM00035121-data.txt"], # 351210 ORENBURG  51.795786 55.456744 RSM00035121
  # ['341220',"RSM00034122-data.txt"], # 341220 CHERTOVITSKOYE  51.814211 39.229589 RSM00034122
  ['307100',"RSM00030715-data.txt"], # 307100 IRKUTSK 52.268028 104.388975  RSM00030715
  # ['296340',"RSM00029634-data.txt"], # 296340 TOLMACHEVO  55.012622 82.650656 RSM00029634
  ['303090',"RSM00030309-data.txt"], # 303090 BRATSK  56.370556 101.698331  RSM00030309
  ['282250',"RSM00028225-data.txt"], # 282250 BOLSHOYE SAVINO 57.914517 56.021214 RSM00028225
  ['259130',"RSM00025913-data.txt"], # 259130 SOKOL 59.910989 150.720439  RSM00025913
  ['234710',"RSM00023955-data.txt"], # 234710 NIZHNEVARTOVSK  60.949272 76.483617 RSM00023955
  ['218230',"RSM00024959-data.txt"], # 218230 YAKUTSK 62.09325  129.770672  RSM00024959
  ['225500',"RSM00022543-data.txt"], # 225500 TALAGI  64.600281 40.716667 RSM00022543
  ['221130',"RSM00022217-data.txt"], # 221130 MURMANSK  68.781672 32.750822 RSM00022217

  # ## ['298380',"RSM00029839-data.txt"], # 298380 BARNAUL 53.363775 83.538533 RSM00029839
  # ## ['239330',"RSM00023933-data.txt"], # 239330 KHANTY MANSIYSK 61.028479 69.086067 RSM00023933
  # '261148', # Минск
  # ['307580',"RSM00030758-data.txt"], # 307580 KADALA  52.026317 113.305556  RSM00030758
  # ['271990',"RSM00027199-data.txt"], # KIROV
  # '260630', # Левашово
  # '262580', # Псков
  # '370540', # Мин воды
  # '370010', #витязево
  # '339460', #SIMFEROPOL
  # '345600', #GUMRAK
  # '274590', #STRIGINO
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


# ['106190', 'GMM00010618-data.txt' ],	# IDAR-OBERSTEIN
#### ['107380', 'GMM00010739-data.txt' ],	# STUTTGART/SCHNARRENBERG мало данных
# ['115180', 'EZM00011520-data.txt' ],	# PRAHA-LIBUS
# ['119340', 'LOM00011952-data.txt' ],	# POPRAD-GANOVCE
# ['132720', 'RIM00013275-data.txt' ],	# BEOGRAD/KOSUTNJAK
# ['260380', 'ENM00026038-data.txt' ],	# TALLINN-HARKU
# ['484315', 'THM00048357-data.txt' ],	# NAKHON PHANOM
# ['485650', 'THM00048565-data.txt' ],	# PHUKET AIRPORT
# ['034185', 'UKM00003354-data.txt' ],	# NOTTINGHAM
# ['038090', 'UKM00003808-data.txt' ],	# CAMBORNE
# ['071460', 'FRM00007145-data.txt' ],	# TRAPPES
# ['076460', 'FRM00007645-data.txt' ],	# NIMES-COURBESSAC
# ['170310', 'TUM00017030-data.txt' ],	# SAMSUN
# ['170630', 'TUM00017064-data.txt' ],	# ISTANBUL BOLGE (KARTAL)
# ['171310', 'TUM00017130-data.txt' ],	# ANKARA/CENTRAL
# ['172410', 'TUM00017240-data.txt' ],	# ISPARTA
# ['401800', 'ISM00040179-data.txt' ],	# BET DAGAN
### ['407060', 'IRM00040706-data.txt' ],	# TABRIZ
### ['407540', 'IRM00040754-data.txt' ],	# TEHRAN-MEHRABAD
### ['407660', 'IRM00040841-data.txt' ],	# KERMAN
# 
# 
### ['408000', 'IRM00040800-data.txt' ],	# ESFAHAN
# ['408110', 'IRM00040811-data.txt' ],	# AHWAZ
# ['416750', 'PKM00041675-data.txt' ],	# MULTAN
# ['419220', 'BGM00041923-data.txt' ],	# DHAKA
# ['423390', 'INM00042339-data.txt' ],	# JODHPUR
# ['423690', 'INM00042369-data.txt' ],	# LUCKNOW/AMAUSI
# ['426470', 'INM00042647-data.txt' ],	# AHMADABAD
# ['427240', 'INM00042724-data.txt' ],	# AGARTALA
# ['601560', 'MOM00060155-data.txt' ],	# CASABLANCA
# ['623660', 'EGM00062378-data.txt' ],	# HELWAN


]


# делаем цикл по станциям
for stantion in st_list:
  # 
  # создаем класс
  # 
  trainer = ForecastCONVAeroThunder()

  # 
  # Настриваем тренировочную базу
  # 
  trainer.set_stantion( stantion[0] )  \
         .set_dt( dt_cur )             \
         .try_fill_stantion_name()     \
         .set_dt_predicted( dt_pred )  \
         .set_filename( stantion[1] )  \
         .set_timestep_predicted( n_timesteps_pred )   \
         .set_params( params )         \
         .set_timesteps( n_timesteps ) \
         .set_backtimedelta( n_backtime_delta ) \
         .set_backtime( n_backtime )   \
         .set_inputs(n_inputs)         \
         .set_batch_size(batch_size)   \
         .set_plot(True)               \
         .set_save(True)               \
         .set_load(True)               \
         .set_shuffle(True)            \
         .set_processes(9)             \
         .set_dbtype( 'isd' )          \
         .set_epochs( num_epochs )     \
         .set_model_name("OYAP_THUNDER")    

  # 
  # тренируем или используем загруженные модели
  # 

  
  trainer.load()    \
         .parse_aero() \
         .load_aero() \
         .merge_air_and_surf("fact") \
         .train()
         
  
  # для прогноза
  # trainer.load_model(params[0])
  # trainer.fillfromdt_predict(dt_pred,n_timesteps_pred) \
  #        .load_predicted() \
  #        .parse_aero() \
  #        .load_aero() \
  #        .merge_air_and_surf(_type="pred") \
  #        .mkbatch_predict( params[0] ) 
  # trainer.X_train = trainer.X_batch
  # trainer.y_train = trainer.Y_batch
  # trainer.X_test  = trainer.X_batch
  # trainer.y_test  = trainer.Y_batch
  # trainer.scale()            \
  #        .plot_predict(show=True,param=params[0],save=True)
  # trainer.plot_corrmatrix()
  # exit() 

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