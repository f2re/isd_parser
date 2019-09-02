# -*- coding: utf-8 -*-
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

import json

import pprint

from collections import defaultdict
from collections import Counter

# подключаем библиотеку подключения к монго
import mongoconnect as mc

from stantionclass import Stantion

from ilistweather import IListWeather

#
# многопроцессорность
#

import multiprocessing as mp


# 
# 
#  Считываем погоду по дням
#  

def get_weather_by_day( date, db, collection, ip, port ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)

  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=0,minute=0,second=59 )
  # ищем станции рядом 
  query = { "date": { "$gte":from_date , "$lte":to_date }, "station.country":"RS"  }
  
  res      = db.find(query)
  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    weather.append( st.fromMongoISDFormat(doc) )
  print(count)
  # возвращаем станции
  return weather

# 
# диапазон дат 
# 
def daterange(start_date, end_date):
  for n in range(int ((end_date - start_date).days)):
    yield start_date + timedelta(n)


# 
# Запрос для базы которая ISD (распарсена от НОАА)
# 
def get_weather_on_ISDstation( date, st, offset, step, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )

  # ищем станции рядом 
  query = { "date": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "station.stantion":st }

  res = get_query( db, collection, ip, port, query )

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoISDFormat(doc)
    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list

# 
# Запрос для базы которая наша
# 
def get_weather_on_REPORTstation( date, st, offset, step, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )

  # ищем станции рядом 
  query = { "dt": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "station":st }
  res = get_query( db, collection, ip, port, query )

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoFormat(doc)
    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list

# 
# получаем погоду на станции 
# @st   - станция
# @offset - смещение в днях от начала
# @step - шаг в днях
# 
def get_query( db, collection, ip, port, query={} ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)
  print(query)
  # делаем запрос в базу
  res          = db.find(query)
  return res

# 
# Подготавливаем данные к обучению на рекурентной сети
# надо разбить массив и сделать его трехмерным
# так как есть шаг по времени
# 
def prepare_sample(data, length):
  samples = list()
  n       = len(data)
  # print(data.shape)
  for i in range(0,n,length):
    sample = data[i:i+length] 
    if len(sample)==length:
      samples.append(sample)
  # print(len(samples)*length)
  # print(data[:len(samples)*length])
  ret = np.array(data[:len(samples)*length]).reshape(len(samples),length,len(data[0]))

  # print(ret)
  # print( ret.shape )
  return ret

# 
# Подготавливаем данные к обучению на рекурентной сети
# надо разбить массив и сделать его трехмерным
# так как есть шаг по времени
# 
def prepare_XY_sample(dataX,dataY, length):
  samples = list()
  Y       = list()
  n       = len(dataX)
  # print(dataX.shape)
  count = 0
  for i in range(0,n,1):
    sample = dataX[i:i+length] 
    if len(sample)==length and i<n:
      samples.append(sample)
      Y.append(dataY[i])
      count+=1
  retX = np.array(samples).reshape(len(samples),length,len(dataX[0]))
  return retX,np.array(Y)

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


# from __future__ import print_function, division
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.backend as K

num_epochs                = 6
total_series_length       = 50000
truncated_backprop_length = 15
state_size                = 4
num_classes               = 2
echo_step                 = 3
batch_size                = 5
num_batches               = total_series_length//batch_size//truncated_backprop_length

# входные параметры
# количество входных параметров
n_inputs                  = 11
# шаг по времени (дни)
n_timesteps               = 700
# количество измерений назад
# еще это может быть length
n_backtime                = 3


# 
# Обучаем и тренируем модель на данных и з БД
# параметры перечислены в виде констант
# Модель LSTM - рекуррентная модель нейросети, которая испольузет шаг заблаговременности n_backtime
# 
# @plot - рисовать ли графики и выводить ли отладочную информацию
# @save - сохраняем модель в файл
# @load - загружаем модель из файла
# @stantion - станция
# 
# return model, scalerX, scalerY
# Возвращаются масштабаторы и модель
# 
def train_model( stantion="219310", plot=False, save=True, load=False ):

  # определяем входные параметры
  X_batch = []
  Y_batch = []

  # 
  # Получаем погоду из базы
  # 
  # 219310 - Юбилейная (21931)
  # 340560 - Ртищево (34056)
  # 
  weather_list = get_weather_on_ISDstation(date=dt_cur,st=stantion,offset=0,step=n_timesteps,db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port)

  # 
  # Создаем набор с данными из базы
  # отбираем набор параметров для обучения нейросети
  # 
  i     = 0
  for item in weather_list.get_all():
    item.onlyFloatOn()
    X_batch.append( [ item.get_T(), 
                      item.get_P(), 
                      item.get_N(), 
                      item.get_dd(), 
                      item.get_VV(), 
                      item.get_ff(), 
                      item.get_H(),                       
                      item.get_RR(),                       
                      item.get_hour(), 
                      item.get_date().day,
                      item.get_date().month ] )
    if i>0 :
      Y_batch.append( item.get_T() )
    i+=1

  # 
  # Уменьшаем массив исходных данных на 1, так как Y на 1 опережает X
  # 
  X_batch=np.array(X_batch[:-1])

  # 
  # Разбиваем выборку на тестовую и обучающую выборку
  # 
  X_train, X_test, y_train, y_test = train_test_split(X_batch, Y_batch, test_size=0.3, random_state=0, shuffle=False)

  # 
  ## Масштабируем данные 
  # 
  # normalize the dataset
  scalerX  = MinMaxScaler()
  scalerY  = MinMaxScaler()

  scalerX = scalerX.fit(X_train)
  X_train = scalerX.transform(X_train)
  X_test  = scalerX.transform(X_test)

  y_train = np.array(y_train).reshape(-1,1)
  y_test  = np.array(y_test).reshape(-1,1)

  scalerY = scalerY.fit(y_train)
  y_train = scalerY.transform(y_train)
  y_test  = scalerY.transform(y_test)

  # 
  # приводим массив данных для рекурентной нейросети
  # здесь мы разбиваем испытуемую выборку на шаги по времени со смещением
  # а ответы (Y) записываем как следующий за тестовой выборкой по сроку
  # 
  X_train , y_train = prepare_XY_sample(X_train,y_train,n_backtime)
  X_test  , y_test  = prepare_XY_sample(X_test,y_test,n_backtime)


  history = False

  # 
  # Загружаем модель
  # 
  if load:
    # load json and create model
    json_file = open("model_"+str(stantion)+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_"+str(stantion)+".h5")
    print("Loaded model from disk")

  else:
    #
    ## Define the network
    # 
    model = keras.Sequential([
        keras.layers.LSTM(n_inputs,input_shape=(1,n_backtime,X_train.shape[2]), 
                          return_sequences=True,
                          batch_input_shape=(1,n_backtime,X_train.shape[2]) ),
        # keras.layers.LSTM(n_inputs, return_sequences=True, stateful=True),
        # keras.layers.LSTM(n_inputs*2, return_sequences=True, stateful=True),
        keras.layers.LSTM(n_inputs, stateful=False,return_sequences=False),
        # keras.layers.LSTM(6, stateful=True,return_sequences=True),
        # keras.layers.LSTM(6, stateful=False),
        keras.layers.Dense(1)
    ])

    # 
    ## Compile model 
    # 
    model.compile(optimizer = 'adam',
                  loss    = 'mean_squared_error',
                  metrics = ['accuracy'])

    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=1,validation_data=(X_test, y_test), verbose=2, shuffle=False)
  
  # 
  # Сохраняем обученную модель на диск
  # 
  if save:
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_"+str(stantion)+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_"+str(stantion)+".h5")
    print("Saved model at "+str(stantion)+" stantion to disk")

  if plot:
    # plot history
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    if history is not False:
      ax1.plot(history.history['loss'], label='train')
      ax1.plot(history.history['val_loss'], label='test')
    ax1.set(title='Loss errors', ylabel='Loss')
    ax1.legend()

    predicted = np.array(list(map( lambda x: scalerY.inverse_transform( 
                                  model.predict( 
                                    x.reshape(1,n_backtime,X_train.shape[2]) 
                                    ) 
                                  ),  X_test ))).flatten()
    # print(predicted)
    ax2.plot( predicted, label="Predicted T" )
    ax2.plot( scalerY.inverse_transform(y_test), label="Real value of T" )
    ax2.set(title='Temp values', ylabel='T')
    ax2.legend()

    ax3.hist(  predicted-scalerY.inverse_transform(y_test).flatten(), label="loss" )
    ax2.legend()
    # выводим графики
    plt.show()

  # 
  # Возвращаем натренированную модель
  # 
  return model, scalerX, scalerY



#
#
# =========  TRAIN MODEL =========
#
#

# 
# Обучаем модель на данных
# 
model, scalerX, scalerY = train_model(  stantion="340560",  save=False, load=True,  plot=True )

# 
# Загружаем свежие данные из базы
# 
weather_list = get_weather_on_ISDstation( date=dt(2019,5,10), st="340560", offset=0, step=5,
                                          db=mongo_db, collection=mongo_collection, ip=mongo_host, port=mongo_port )
                                       # db='meteodb', collection='meteoreport', 
                                       # ip='10.10.11.120', port=mongo_port )

# 
# Создаем набор с данными из базы
# отбираем набор параметров для обучения нейросети
# 
i     = 0
X_batch=[]
Y_batch=[]
for item in weather_list.get_all():
  item.onlyFloatOn()
  X_batch.append( [ item.get_T(), 
                    item.get_P(), 
                    item.get_N(), 
                    item.get_dd(), 
                    item.get_VV(), 
                    item.get_ff(), 
                    item.get_H(),                       
                    item.get_RR(),                       
                    item.get_hour(), 
                    item.get_date().day,
                    item.get_date().month ] )
  if i>0 :
    Y_batch.append( item.get_T() )
  i+=1

# 
# Уменьшаем массив исходных данных на 1, так как Y на 1 опережает X
# 
X_batch=np.array(X_batch[:-1])

X_batch = scalerX.transform(X_batch)
Y_batch = scalerY.transform(np.array(Y_batch).reshape(-1,1))

X_train , y_train = prepare_XY_sample(X_batch,Y_batch,n_backtime)


predicted = np.array(list(map( lambda x: scalerY.inverse_transform( 
                              model.predict( 
                                x.reshape(1,n_backtime,X_train.shape[2]) 
                                ) 
                              ),  X_train ))).flatten()
fig, (ax1, ax2) = plt.subplots(2)
# print(predicted)
ax1.plot( predicted, label="Predicted T" )
ax1.plot( scalerY.inverse_transform(y_train), label="Real value of T" )
ax1.set(title='Значения температуры по данным, которых не было в обучающей выборке', ylabel='T')
ax1.legend()

ax2.hist(  predicted-scalerY.inverse_transform(y_train).flatten(), label="loss" )
ax2.legend()
# выводим графики
plt.show()

exit() 