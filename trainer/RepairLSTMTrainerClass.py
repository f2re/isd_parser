# -*- coding: utf-8 -*-

# 
# Класс восстановления значений на станции которая находится по соседству
# алгоритм работы:
# задаем номер станции, которую надо восстановить
# ищем станции, которые правее (или левее ) в радиусе 200 км 
# забираем данные по станциям
# обучаем модель
# 

from .RepairDataTrainerClass import RepairDataTrainerClass
from .dbfunctions import *

from   datetime import datetime as dt
from   datetime import date, timedelta
import time

from .stantionclass import Stantion

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import multiprocessing, logging
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from geopy.distance import geodesic

import keras
import keras.backend as K

class RepairLSTMTrainerClass(RepairDataTrainerClass):

  def __init__(self):
    super(RepairLSTMTrainerClass, self).__init__()

    return

  def train(self):
    for param in self._params:
      self.mkbatch( param ) \
         .split_batch() \
         .scale() 

      if self._model is None:
         self.set_default_model()

      self.fit() \
         .save( param ) \
         .plot(param=param) \
         .predict(param=param)
    return self


  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self):
    ## Define the network
    # 
    x_sz = (self.X_train.shape[0]//self._batch_size)*self._batch_size
    self.X_train = self.X_train[0:x_sz]
    self.y_train = self.y_train[0:x_sz]

    x_sz = (self.X_test.shape[0]//self._batch_size)*self._batch_size
    self.X_test = self.X_test[0:x_sz]
    self.y_test = self.y_test[0:x_sz]

    self.set_model( keras.Sequential([
        # keras.layers.Embedding(self.X_train.shape[2], 39,input_length = self.X_train.shape[1], dropout = 0.2),
        keras.layers.LSTM(self.X_train.shape[2], 
                          input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                          return_sequences  = False,
                          stateful          = False,
                          batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) ),
        # keras.layers.LSTM(10, stateful=False,return_sequences=False), #SKO 4.475120590097023 Median 2.1805897712707503
        # keras.layers.LSTM(20, stateful=False,return_sequences=False),
        # keras.layers.Dense(self.X_train.shape[2]*2),
        # keras.layers.Dense(3, activation="relu"), # SKO 4.526254279907294 Median 2.445237445831298
        # keras.layers.Dense(5, activation="relu"), # SKO 4.484138405526309 Median 1.6294190406799318
        # keras.layers.Dense(20, activation="relu"), # SKO 4.51334918486406 Median 1.9356906890869134
        # keras.layers.Dense(300,activation="relu"),
        keras.layers.Dense(50,activation="relu"),
        keras.layers.Dense(1)
    ]) )
    # 
    ## Compile model 
    # 
    self._model.compile(
                  optimizer = 'adagrad',
                  # optimizer = 'sgd',
                  # optimizer = 'adam',
                  loss    = 'mean_absolute_error',
                  # loss    = 'mean_squared_error',
                  metrics = ['accuracy'])
    print(self._model.summary())
    return self

  # 
  # Делаем батчи
  # @param - параметр по которому будем прогнозировать (Y)
  # @wlist - список IWeatherList
  # 
  def makebatch(self,param,wlist, fromdt, todt):
    _X_batch = []
    _Y_batch = []

    i         = 0
    
    if self._results is not None:
      results = self._results
    else:
      self._wlist_cur.makehash()
      results = self.get_array_mp( timedelta( hours=3 ), fromdt, todt )

    # exit()
    # 
    # здесь это массив IListweather
    # с шагом по времени
    # 
    for arraypool in results:
      # print(arraypool)
      # 
      # забираем последнюю станцию - это та, по которой нам нужны прогнощзы
      # значения на этой стании будут использоваться дл формирования ответов
      # 
      # Y_stantion   = stantionlist.pop()
      Y_stantion = arraypool[0].get_all()[-1]
      
      # 
      # если мы жесто хотим чтобы анализировались только 
      # параметры, когда есть правильный ответ
      # 
      if self._only_with_value:
        if Y_stantion.get_byparam(param)==0:
          continue

      # обнуляем массив
      arr = np.array([])

      # 
      # проходимся по сплиту из 3 времен назад
      # 
      for itemarray in arraypool:

        startdate = None
        # 
        # Проходимяс и рассчитываем время восхода/захода
        # 
        sunrises = np.array([])
        for item in itemarray.get_all():

          if item.get_stantion()!='000000':
            if startdate is None:
              startdate = item.get_date()

            item.calcSun()
            sunrises = np.append( sunrises,item.get_sunAll()['sunrise'] )
            sunrises = np.append( sunrises,item.get_sunAll()['sunset'] )

        # 
        # это весь списко станций без последней (позже ее заберем)
        # 
        stantionlist = itemarray.get_all()[:-1]
        # print("stantion list wothout last: ",stantionlist)
        
        # формируем массив значений параметра по станциям
        # важно соблюдать порядок станций (чтобы не менялся)
        itemlist = [ [item.get_T(),item.get_P(),item.get_Td(),item.get_VV(),item.get_ff(),item.get_dd(),item.get_N(),item.get_H()] for item in stantionlist ]
        
        if len(itemlist)>0 and startdate is not None:
          # 
          # здесь добавляем все остальные значения на станциях
          # 
          res = np.append( np.array(itemlist).flatten(), 
                                np.asarray([
                                          (startdate.hour), 
                                          (startdate.day), 
                                          (startdate.month)])
                           )
          if len(arr)==0:
            arr = np.array([res])
          else:
            arr = np.append( arr, np.array([res]), axis=0 )
        i+=1
        # endfor

      if len(arr)==self.n_backtime:
        _X_batch.append( arr )
        # 
        # последнее значение забираем для ответа
        # 
        _Y_batch.append( Y_stantion.get_byparam(param) )

    return _X_batch, _Y_batch


  # 
  # Получаем массив (ищем по дате)
  # 
  def get_itemarray(self,dt):
    # 
    # забираем станции за эту дату
    # 
    itemarray = self._wlist_cur.get_items_by_date_st_and_timeback(dt,Stantion,self.n_backtime,self.n_backtime_delta)
    return itemarray


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

    shp = np.shape(self.X_train)
    # print(np.shape( np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ) ))

    self._scalerX = self._scalerX.fit(np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ))
    self.X_train  = np.reshape( 
                      self._scalerX.transform(np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ))
                      , (shp[0], shp[1], shp[2])
                    )

    _shp = np.shape(self.X_test)
    self.X_test   = np.reshape( 
                      self._scalerX.transform( np.reshape(self.X_test, ( _shp[0]*shp[1], shp[2])  ) )
                      , (_shp[0], shp[1], shp[2])
                    )

    self.y_train = np.array(self.y_train).reshape(-1,1)
    self.y_test  = np.array(self.y_test).reshape(-1,1)

    self._scalerY = self._scalerY.fit(self.y_train)
    self.y_train  = self._scalerY.transform(self.y_train)
    self.y_test   = self._scalerY.transform(self.y_test)
    return self


  # 
  # Рисуем матрицу корреляции на основе данных, которые пришли
  # 
  def plot_corrmatrix(self,param):

    stantions_data = {}
    self._wlist_cur = self._weather_list

    if self._results is not None:
      results = self._results
    else:
      self._wlist_cur.makehash()
      results = self.get_array_mp( timedelta( hours=3 ),self._from_date, self._to_date )

    for arraypool in results:
      # 
      # проходимся по ответу и работаем с данными
      # 
      for itemarray in arraypool:
        for item in itemarray.get_all():
          if item.get_stantion()!='000000':
            # если такая станция уже есть
            if item.get_stantion() not in stantions_data:
              stantions_data[ item.get_stantion() ] = [ item.get_byparam(param) ]
            else:
              stantions_data[ item.get_stantion() ].append( item.get_byparam(param) )

    sframe = {}
    for key in stantions_data.keys():
      sframe[key] = pd.Series(stantions_data[key])
    # 
    # загружаем все в панду
    # 
    df = pd.DataFrame(sframe)

    corr = df.corr().astype(float)

    # plt.fig(figsize=(40,40)) 
    import seaborn as sns
    sns.heatmap(data=corr, annot=True, fmt='2.2f', cmap='Greens')
    plt.title("Матрица корреляций между параметром <"+param+"> на станциях. Станция прогноза "+self._stantion)
    if self._save:
      path = self._model_name+"_"+self._stantion
      plt.savefig(path+"/corr1_"+self._model_name+"_"+param+"_"+str(self._stantion)+".png")
    if self._plot:
      plt.show()

    # sns.pairplot(corr, kind="reg")
    # plt.title("Матрица корреляций между параметром <"+param+"> на станциях. Станция прогноза "+self._stantion)
    # if self._save:
    #   path = self._model_name+"_"+self._stantion
    #   plt.savefig(path+"/corr2_"+self._model_name+"_"+param+"_"+str(self._stantion)+".png")

    if self._plot:
      plt.show()

    return self

  # 
  # Подготавливаем данные для того, чтобы запихнуть в функцию пронозирования
  # надо ли решейпнуть массив или еще чего
  # все здесь
  # используется в функции прогнозирования
  # 
  def prepare_x_to_predict(self,x,shp):
    # print("shape: ", self._batch_size,self.n_backtime,shp[2])
    # print(x.reshape(self._batch_size,self.n_backtime,self.X_train.shape[2]).shape)
    # shp = self.chunk(self.X_train,self._batch_size)
    xx = x.reshape(self._batch_size,self.n_backtime,shp[2])
    # print(xx.shape)
    return xx
