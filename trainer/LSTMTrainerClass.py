# -*- coding: utf-8 -*-

# 
# 
# Класс работы с обучением моделей
# 
# обучаем модели пачками, смотрим параметры для обучения
# сохраняем результаты работы моделей в папки
# сохраняем картинки и графики в папки чтобы сравнить эффективность моделей
# 
# 

import os.path
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from   datetime import datetime as dt
from   datetime import date, timedelta
from   pymongo import MongoClient
from   collections import defaultdict
from   collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# подключаем библиотеку подключения к монго
# import mongoconnect as mc

from stantionclass import Stantion
from ilistweather import IListWeather

from .dbfunctions import *

import keras
import keras.backend as K

class LSTMTrainerClass(object):

  def __init__(self):

    self.mongo={
                'db':         'srcdata',
                'collection': 'meteoisd',
                'host':       'localhost',
                'port':       27017 }

    # текущая дата, с которой начинаем запрос в базу
    self._dt      = dt(2017, 1, 1)

    # 
    # это параметры для запроса данных и проверки адекватности модели
    # 
    self._dt_predicted = dt(2019,1,1)
    self._timestep_predicted = 100
    self._stantion_predicted = ''

    # 
    # стиль графиков
    # 
    self._plotstype = 'seaborn-pastel'

    # 
    # Параметры модели
    # 
    # дни в которых ведется отсчет от базового времени
    self.n_timesteps = 1
    # входные параметры
    # количество входных параметров
    self.n_inputs    = 11
    # количество измерений назад
    # еще это может быть length
    self.n_backtime  = 3
    # какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
    self.n_dropout   = 0.2    
    # количество эпох
    self.num_epochs  = 10
    # 
    # Модель, которая будет работать
    # 
    self._model      = None
    self._model_name = "default"
    # 
    # Масштабаторы
    # 
    self._scalerX = None 
    self._scalerY = None

    # 
    # Рисуем графики
    # 
    self._plot = False
    # 
    # Сохраняем модель
    # 
    self._save = True 
    # 
    # Загружаем модель
    # 
    self._load = False

    # 
    # Станция, по которой считаем
    # 
    self._stantion = ""

    # 
    # Тип базы из которой запрашиваем данные
    # может быть isd и native
    # 
    self._dbtype = 'isd'

    # 
    # Набор параметров по которым будем тренировать
    # если он один - тренируем одну модель
    # 
    self._params = ['T']

    # 
    # параметры разделения выборки на тесты
    # параметры функции train_test_split
    # 
    self._test_size    = 0.3
    self._random_state = 0
    self._shuffle      = False

    # 
    # Батчи данных
    # 
    self.X_batch = []
    self.Y_batch = []

    self.X_train = []
    self.X_test  = [] 
    self.y_train = []
    self.y_test  = []
    # погода из базы
    self._weather_list = []
    self._weather_list_predicted = []
    # даты запроса из базы
    self._from_date = None
    self._to_date   = None
    self._from_date_predicted = None
    self._to_date_predicted   = None
    return

  # 
  # Запускаем обучение модели
  # 
  def train(self):
    for param in self._params:
      self.mkbatch( param ) \
         .split_batch() \
         .scale() \
         .preparesample() \
         .set_default_model()  \
         .fit() \
         .save( param ) \
         .plot(param=param) \
         .predict(param=param)
    return self

  def load(self):
    # 
    # Получаем погоду из базы
    # 
    # 219310 - Юбилейная (21931)
    # 340560 - Ртищево (34056)
    # 
    if self._dbtype=="isd":
       self._weather_list, \
       self._from_date,    \
       self._to_date = get_weather_on_ISDstation(
                                date       = self._dt,
                                st         = self._stantion,
                                offset     = 0,
                                step       = self.n_timesteps,
                                db         = self.mongo['db'], 
                                collection = self.mongo['collection'], 
                                ip         = self.mongo['host'], 
                                port       = self.mongo['port'] )
    else:
      pass
    return self

  # 
  # загружаем данные для прогностических моделей
  # 
  def load_predicted(self):
    self._weather_list_predicted, \
    self._from_date_predicted,    \
    self._to_date_predicted = get_weather_on_ISDstation(
                             date       = self._dt_predicted,
                             st         = self._stantion_predicted,
                             offset     = 0,
                             step       = self._timestep_predicted,
                             db         = self.mongo['db'], 
                             collection = self.mongo['collection'], 
                             ip         = self.mongo['host'], 
                             port       = self.mongo['port'] )
    return self

  # 
  # преобразуем исходные данные в батчи пригодные для обучения
  # 
  def mkbatch(self,param):
    self.X_batch = []
    self.Y_batch = []
    self.X_batch, self.Y_batch = self.makebatch( param, self._weather_list )
    return self

  # 
  # 
  # 
  def mkbatch_predict(self,param):
    self.X_batch = []
    self.Y_batch = []
    self.X_batch, self.Y_batch = self.makebatch( param, self._weather_list_predicted )
    return self

  # 
  # Делаем батчи
  # @param - параметр по которому будем прогнозировать (Y)
  # @wlist - список IWeatherList
  # 
  def makebatch(self,param,wlist):
    _X_batch = []
    _Y_batch = []

    i = 0
    for item in wlist.get_all():
      item.onlyFloatOn()
      _X_batch.append( [  item.get_T(), 
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
        app = None
        if param=="T":  
          app = item.get_T()
        elif param=="P":
          app = item.get_P()
        elif param=="V":
          app = item.get_VV()
        elif param=="dd":
          app = item.get_dd()
        elif param=="ff":
          app = item.get_ff()

        _Y_batch.append( app )
      i+=1

    # 
    # Уменьшаем массив исходных данных на 1, так как Y на 1 опережает X
    # 
    _X_batch=np.array(_X_batch[:-1])
    return _X_batch, _Y_batch

  # 
  # разделяем нашу выборку на тренировочную и проверочную
  # 
  def split_batch(self):
    self.X_train, \
    self.X_test,  \
    self.y_train, \
    self.y_test = train_test_split( self.X_batch, 
                                    self.Y_batch, 
                                    test_size    = self._test_size, 
                                    random_state = self._random_state,
                                    shuffle      = self._shuffle )
    return self


  # 
  # Масштабируем наши исходные данные
  # 
  def scale(self):
    # 
    ## Масштабируем данные 
    # 
    # normalize the dataset
    self._scalerX  = MinMaxScaler()
    self._scalerY  = MinMaxScaler()

    self._scalerX = self._scalerX.fit(self.X_train)
    self.X_train  = self._scalerX.transform(self.X_train)
    self.X_test   = self._scalerX.transform(self.X_test)

    self.y_train = np.array(self.y_train).reshape(-1,1)
    self.y_test  = np.array(self.y_test).reshape(-1,1)

    self._scalerY = self._scalerY.fit(self.y_train)
    self.y_train  = self._scalerY.transform(self.y_train)
    self.y_test   = self._scalerY.transform(self.y_test)
    return self


  # 
  # подготавливаем тренировочные данные к обучению на рекуреннтной сети
  # 
  def preparesample(self):
    # 
    # приводим массив данных для рекурентной нейросети
    # здесь мы разбиваем испытуемую выборку на шаги по времени со смещением
    # а ответы (Y) записываем как следующий за тестовой выборкой по сроку
    # 
    self.X_train , self.y_train = self.prepare_XY_sample( self.X_train, self.y_train, self.n_backtime )
    self.X_test  , self.y_test  = self.prepare_XY_sample( self.X_test,  self.y_test,  self.n_backtime )
    return self

  # 
  # Подготавливаем данные к обучению на рекурентной сети
  # надо разбить массив и сделать его трехмерным
  # так как есть шаг по времени
  # 
  def prepare_XY_sample(self, dataX,dataY, length):
    samples = list()
    Y       = list()
    n       = len(dataX)
    count   = 0
    for i in range(0,n,1):
      sample  = dataX[i:i+length] 
      if len(sample) == length and i<n:
        samples.append(sample)
        Y.append(dataY[i])
        count += 1
    retX = np.array(samples).reshape(len(samples),length,len(dataX[0]))
    return retX,np.array(Y)


  # 
  # Сохраняем модель и результаты в папку
  # 
  def save(self,param):
    if self._save is True:
      # задаем имя папки куда складывать будем модели
      path = self._model_name+"_"+self._stantion
      # 
      # если папки нет, создаем ее
      # 
      if not os.path.exists(path):
        os.makedirs(path)

      plt.rcParams["figure.figsize"]=(25, 10)#Размер картинок

      # serialize model to JSON
      model_json = self._model.to_json()
      with open( path+"/model_"+str(self._stantion)+"_"+param+".json", "w" ) as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      self._model.save_weights(path+"/model_"+str(self._stantion)+"_"+param+".h5")
      print("Saved model <"+self._model_name+"> at <"+str(self._stantion)+"> stantion to disk")

      # 
      # Сохраняем картинки
      # 
      # plot history
      self.plot(show=False,param=param,save=True)
    return self

  # 
  # Рисуем графики и показываем их
  # 
  def plot(self,show=True,param="T",save=False):
    plt.style.use( self._plotstype )
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    if self.history is not False:
      ax1.plot(self.history.history['loss'], label='train')
      ax1.plot(self.history.history['val_loss'], label='test')
    ax1.set(title='Loss errors', ylabel='Loss')
    ax1.legend()

    predicted = np.array(list(map( lambda x: self._scalerY.inverse_transform( 
                                  self._model.predict( 
                                    x.reshape(1,self.n_backtime,self.X_train.shape[2]) 
                                    ) 
                                  ),  self.X_test ))).flatten()
    ax2.plot( predicted, label="Predicted " )
    ax2.plot( self._scalerY.inverse_transform(self.y_test), label="Real value of "+param )
    ax2.set(title=param+' values '+(self._from_date.strftime("%Y.%m.%d"))+"-"+(self._to_date.strftime("%Y.%m.%d")), ylabel=param )
    ax2.legend()

    ax3.hist(  predicted-self._scalerY.inverse_transform(self.y_test).flatten(), label="loss" )
    ax2.legend()

    # 
    # если в настройках установлено что показываем график
    # и мы вызываем метод не из кода
    # то показываем график
    # 
    # if self._plot and show:
      # plt.show()
    if save:
      path = self._model_name+"_"+self._stantion
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_train.png")
    return self

  # 
  # Рисуем график для верификации прогнозов
  # 
  def plot_predict(self,show=True,param="T",save=False):
    plt.style.use( self._plotstype )
    fig, (ax1, ax2) = plt.subplots(2)
    predicted = np.array(list(map( lambda x: self._scalerY.inverse_transform( 
                          self._model.predict( 
                            x.reshape(1,self.n_backtime,self.X_train.shape[2]) 
                            ) 
                          ),  self.X_train ))).flatten()
        
    ax1.plot( predicted, label="Predicted "+param )
    ax1.plot( self._scalerY.inverse_transform(self.y_train), label="Real value of "+param )
    ax1.set( title  = 'Значения температуры по данным, которых не было в обучающей выборке'+(self._from_date_predicted.strftime("%Y.%m.%d"))+"-"+(self._to_date_predicted.strftime("%Y.%m.%d")),
             ylabel = param )
    ax1.legend()

    ax2.hist(  predicted-self._scalerY.inverse_transform(self.y_train).flatten(), label="loss" )
    ax2.legend()
    # выводим графики
    # plt.show()
    if save:
      path = self._model_name+"_"+self._stantion
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_predict.png")
    return self

  # 
  # Обучаем модель
  # 
  def fit(self):
    self.history = self._model.fit( self.X_train, 
                                    self.y_train, 
                                    epochs          = self.num_epochs, 
                                    batch_size      = 1,
                                    validation_data = (self.X_test, self.y_test), 
                                    verbose         = 2, 
                                    shuffle         = False )
    return self

  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self):
    ## Define the network
    # 
    self.set_model( keras.Sequential([
        keras.layers.LSTM(self.n_inputs, input_shape=(1,self.n_backtime,self.X_train.shape[2]), 
                          return_sequences=True,
                          stateful=False,
                          batch_input_shape=(1,self.n_backtime,self.X_train.shape[2]) ),
        keras.layers.LSTM(self.n_inputs, stateful=False,return_sequences=False),
        keras.layers.Dense(1)
    ]) )

    # 
    ## Compile model 
    # 
    self._model.compile(optimizer = 'adam',
                  loss    = 'mean_squared_error',
                  metrics = ['accuracy'])

    return self


  # 
  # Теперь проверяем модель на адекватность
  # будем проверять данными, про кторые модель не знает
  # и пробуем спрогнозировать значения
  # 
  def predict(self,param):

    self.load_predicted()   \
        .mkbatch_predict( param ) \
        .split_batch()      \
        .scale()            \
        .preparesample()    \
        .plot_predict(show=False,param=param,save=True)

    return self

  # 
  # ========= SETTERS ===========
  # 

  def set_params(self,arr):
    self._params=arr
    return self

  # 
  # устанав
  # 
  def set_param(self,par):
    self._params=[par]
    return self

  def set_model(self,val):
    self._model = val
    return self

  def set_scalerX(self,val):
    self._scalerX = val
    return self

  def set_scalerY(self,val):
    self._scalerY = val
    return self

  def set_plot(self,val):
    self._plot = val
    return self

  def set_save(self,val):
    self._save = val
    return self

  def set_load(self,val):
    self._load = val
    return self

  def set_stantion(self,val):
    self._stantion = val
    self._stantion_predicted = val
    return self

  def set_stantion_predicted(self,val):
    self._stantion_predicted = val
    return self

  def set_dbtype(self,val):
    self._dbtype = val
    return self

  def set_epochs(self,val):
    self.num_epochs = val
    return self

  def set_dt(self,val):
    self._dt = val
    return self

  def set_inputs(self,val):
    self.n_inputs = val
    return self

  def set_timesteps(self,val):
    self.n_timesteps = val
    return self

  def set_dt_predicted(self,val):
    self._dt_predicted=value
    return self

  def set_timestep_predicted(self,val):
    self._timestep_predicted=val
    return self


  # 
  # устанавливаем параметры соединения с базой
  # 
  def set_mongo(self,db,collection,host,port):
    self.mongo={
                'db':         db,
                'collection': collection,
                'host':       host,
                'port':       port}
    return self

  # 
  # ========= GETTERS ===========
  # 
  
  def get_model(self):
    return self._model

  def get_scalerX(self):
    return self._scalerX

  def get_scalerY(self):
    return self._scalerY

  def get_plot(self):
    return self._plot

  def get_save(self):
    return self._save

  def get_load(self):
    return self._load

  def get_stantion(self):
    return self._stantion

  def get_dbtype(self):
    return self._dbtype

  def get_params(self):
    return self._params
