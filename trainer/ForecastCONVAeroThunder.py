# -*- coding: utf-8 -*-

# 
# Класс восстановления значений на станции которая находится по соседству
# алгоритм работы:
# задаем номер станции, которую надо восстановить
# ищем станции, которые правее (или левее ) в радиусе 200 км 
# забираем данные по станциям
# обучаем модель
# 

from .LSTMTrainerClass import LSTMTrainerClass
from .dbfunctions import *
from .forecast import *

from   datetime import datetime as dt
from   datetime import date, timedelta
import time
from calendar import timegm

from .stantionclass import Stantion

import os.path

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pprint import pprint

from   trainer.sun import Sun

import multiprocessing, logging
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

from geopy.distance import geodesic

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import Input, Dense, BatchNormalization,Conv1D,MaxPooling1D,Flatten, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout, LSTM
from keras.optimizers import SGD
from keras.models import Model

from keras.callbacks import TensorBoard

class ForecastCONVAeroThunder(LSTMTrainerClass):

  def __init__(self):
    super(ForecastCONVAeroThunder, self).__init__()

    return

  def train(self):
    for param in self._params:
      self.mkbatch( param ) \
         .split_batch() \
         .scale(True) 

      if self._model is None:
         self.set_default_model()

      self.fit() \
         .save( param ) \
         .plot(param=param) \
         .predict(param=param)
    return self


  # 
  # Теперь проверяем модель на адекватность
  # будем проверять данными, про кторые модель не знает
  # и пробуем спрогнозировать значения
  # 
  def predict(self,param):

    self.load_predicted()   \
        .merge_air_and_surf(_type="pred") \
        .mkbatch_predict( param )

    # наполняем набор сразу всей выборкой
    self.X_train = self.X_batch
    self.y_train = self.Y_batch

    self.scale()           \
        .plot_predict(show=self._plot,param=param,save=self._save)
        # .split_batch()      \

    return self

  # 
  # Делаем батчи
  # @param - параметр по которому будем прогнозировать (Y)
  # @wlist - список IWeatherList
  # 
  def makebatch(self,param,wlist, fromdt, todt):
    _X_batch = []
    _Y_batch = []
    # имя файла для сохранения/восстановления
    path = self.get_path()
    filename = path+'/prepared_'+param+'_'+fromdt.strftime('%Y%m%d')+'_to_'+todt.strftime('%Y%m%d')+'.dump'
    print(filename)
    # print(self._load)
    if self._load is True:
      self.try_mkdir()
      if os.path.exists(filename):  
        file = open(filename,'rb')
        # print(pickle.load(file))
        _X_batch, _Y_batch = pickle.load(file)

        # 
        # первое значение у Y это если ОЯП наблюдается
        # второе - если ОЯП не наблюдается
        # [True, False] - наблюдается
        # [False, True] - ненаблюдается
        # 
        # print(_Y_batch)
        # new_Y_batch = []
        # for _i in _Y_batch:
        #   ret = [1,0]
        #   if not _i[0]:
        #     ret = [0,1]
        #   _i[0] = ret
        #   new_Y_batch.append(_i)

        # _Y_batch = new_Y_batch
        newbatch = [] 
        for item in _X_batch:
          newbatch.append(item[0])

        df = pd.DataFrame( newbatch )
        # df = pd.DataFrame( _X_batch )
        df.to_csv( path+"/_x_values.csv",sep=";" )
        df = pd.DataFrame( _Y_batch )
        df.to_csv( path+"/_y_values.csv",sep=";" )

        file.close()

        return _X_batch, _Y_batch

    self._wlist_cur.makehash()
    results = self.get_array_mp( timedelta( minutes=5 ), fromdt, todt )

    # 
    # теперь надо получить набор данных с определенной заблаговременностью
    # 
    # здесь это массив 
    # [ [X], [Y] ]
    # 
    for arraypool in results:
      if arraypool is not False and arraypool is not None:
        # 
        # это массив с ответами
        # 
        _Y = arraypool[1]

        # 
        # проходимся по всем данным
        # 
        i     = 0
        x_arr = []
        for _X in arraypool[0]:
          if _X is not None and _X is not False:
            _X.onlyFloatOn()
            _X.calcSun()

            
            
            # 
            # отсутпаем назад на заданную дельту
            # 
            x_arr.append( [ _X.get_T(),
                            _X.get_Td(),
                            _X.get_P(), 
                            _X.get_N(), 
                            _X.get_dd(), 
                            _X.get_VV(), 
                            _X.get_ff(), 
                            _X.get_H(),
                            _X.get_WW(),
                            _X.get_RR(),

                            _X.get_T_850(  ),
                            _X.get_D_850(  ),
                            _X.get_dd_850( ),
                            _X.get_ff_850( ),
                            _X.get_T_925(  ),
                            _X.get_D_925(  ),
                            _X.get_dd_925( ),
                            _X.get_ff_925( ),
                            _X.get_MRi(    ),
                            _X.get_FSI(    ),

                            _X.get_hourmin(),                        
                            _X.get_sunAll()['sunrise'],
                            _X.get_sunAll()['sunset'],
                            _X.get_date().day,
                            _X.get_date().month ] )
          i+=1

        _X_batch.append( x_arr )
        _Y_batch.append( _Y )

    # 
    # сохраняем параметры
    # 
    if self.save:
      self.try_mkdir()
      file = open(filename,'wb')
      pickle.dump( [_X_batch, _Y_batch], file )
      file.close()

    return _X_batch, _Y_batch



  # 
  # делаем поиск по списку данных по времени 
  # @delta - шаг по времени
  # 
  def get_array_mp(self, delta, fromdt, todt):

    # 
    # делаем массив дат, по окторым будем искать
    # 
    dates  = []
    # счетчик назад чтобы пропустить определенное количество дат
    downcount = self.n_backtime
    for startdate in self._wlist_cur.get_hash():
      # уменьшаем счетчик и пропускаем даты в списке
      # print(type(startdate),startdate)
      if downcount>0:
        downcount-=1
        continue
      _dt = dt.utcfromtimestamp(startdate)
      dates.append(_dt)

    pool =  ThreadPool(processes=self._processes)

    # 
    # запускаем мультипроцессорные расчеты
    # 
    results = pool.map(self.get_itemarray,dates) 
    pool.close()
    # pool.terminate()
    # pool.join()
    self._results = results

    return results

  # 
  # Получаем массив (ищем по дате)
  # 
  def get_itemarray(self,_dt):
    # 
    # формируем массив со значениями метеопараметров
    # 
    
    # выводим значения, где есть явления
    # for it in self._wlist_cur.get_all():
    #   if it.get_WW() != '':
    #     print(it)
    #     
    # здесь item содержит два массива
    # [ [..X..],[..Y..] ]
    # один с X - входными параметрами
    # второй с Y - ответами 
    #     
    # print(_dt,int(timegm(_dt.timetuple())))
    # items = self._wlist_cur.get_items_by_oyap(_dt,self.n_backtime,self.n_backtime_delta,self.n_forecastdelta)
    items = self._wlist_cur.get_items_by_oyap(_dt,self.n_backtime,self.n_backtime_delta,True)

    return items


  # 
  # Масштабируем наши исходные данные
  # @fit - тренируем ли или нет на новых данных
  # 
  def scale(self, fit=False):
    # 
    ## Масштабируем данные 
    # 
    # normalize the dataset
    if fit:
      self._scalerX  = MinMaxScaler()
      self._scalerY  = [ LabelEncoder(),MinMaxScaler(),MinMaxScaler() ]

    shp = np.shape(self.X_train)

    # print(shp)

    # self._scalerX = self._scalerX.fit(self.X_train)
    # self.X_train  = self._scalerX.transform(self.X_train)
    # self.X_test   = self._scalerX.transform(self.X_test)

    if fit:
      self._scalerX = self._scalerX.fit(np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ))

    # print(self.X_train)

    self.X_train  = np.reshape( 
                      self._scalerX.transform(np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ))
                      , (shp[0], shp[1], shp[2])
                    )

    _shp = np.shape(self.X_test)
    self.X_test   = np.reshape( 
                      self._scalerX.transform( np.reshape(self.X_test, ( _shp[0]*shp[1], shp[2])  ) )
                      , (_shp[0], shp[1], shp[2])
                    )

    x_sz = (self.X_train.shape[0]//self._batch_size)*self._batch_size
    self.X_train = self.X_train[0:x_sz]
    self.y_train = self.y_train[0:x_sz]

    x_sz = (self.X_test.shape[0]//self._batch_size)*self._batch_size
    self.X_test = self.X_test[0:x_sz]
    self.y_test = self.y_test[0:x_sz]

    # 
    # разбюираем ответы по переменным, чтобы удобно было обучать модель
    # 
    self.y_train_1 = []
    self.y_train_2 = []
    self.y_train_3 = []
    for _ in self.y_train:
      self.y_train_1.append( _[0] )
      self.y_train_2.append( _[1] )
      self.y_train_3.append( _[2] )

    if fit:
      # self._scalerY[0] = self._scalerY[0].fit(np.reshape(self.y_train_1, (-1,1) ) )
      # self._scalerY[0] = self._scalerY[0].fit((self.y_train_1) )
      self._scalerY[1] = self._scalerY[1].fit(np.reshape(self.y_train_2, (-1,1) ) )
      self._scalerY[2] = self._scalerY[2].fit(np.reshape(self.y_train_3, (-1,1) ) )

    # print( ((self.y_train_1 )) )
    # print( np.shape((self.y_train_1 )) )
    # self.y_train_1 = self._scalerY[0].transform(np.reshape(self.y_train_1, (-1,1) ))
    # self.y_train_1 = self._scalerY[0].transform((self.y_train_1))
    self.y_train_2 = self._scalerY[1].transform(np.reshape(self.y_train_2, (-1,1) ))
    self.y_train_3 = self._scalerY[2].transform(np.reshape(self.y_train_3, (-1,1) ))

    # print('train',self.y_train_1)

    self.y_test_1 = []
    self.y_test_2 = []
    self.y_test_3 = []
    for _ in self.y_test:
      self.y_test_1.append( _[0] )
      self.y_test_2.append( _[1] )
      self.y_test_3.append( _[2] )

    # self.y_test_1 = self._scalerY[0].transform(np.reshape(self.y_test_1, (-1,1) ))
    # self.y_test_1 = self._scalerY[0].transform((self.y_test_1))
    self.y_test_2 = self._scalerY[1].transform(np.reshape(self.y_test_2, (-1,1) ))
    self.y_test_3 = self._scalerY[2].transform(np.reshape(self.y_test_3, (-1,1) ))

    # print('test',self.y_test_1)

    return self

 # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self,weights=None):
    ## Define the network
    # 
    a            = Input( batch_shape=(self._batch_size, self.n_backtime, self.X_train.shape[2]) )
    # b            = Input( batch_shape=(self.n_inputs*3) )
    # print(  )
    # input_layer  = LSTM(
    #                     units=self.X_train.shape[2], 
    #                     input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
    #                     return_sequences  = False,
    #                     stateful          = True,
    #                     batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
                        # units=(self._batch_size, self.n_backtime, self.X_train.shape[2]),
                        # units=self.X_train.shape[2],
                       # return_sequences  = False,
                       # stateful          = False )(a)
    # input_layer2 = LSTM((self.n_inputs*3), 
    #                    stateful=False,
    #                    return_sequences=False )(input_layer)
    # _  = Dense(units=self.n_inputs*5,activation="sigmoid")(input_layer)
    input_layer = Conv1D(20,
                           6,
                           padding='valid',
                           activation='relu',
                           batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
    input_layer = Conv1D(8,
                           6,
                           activation='sigmoid', )(input_layer)
    # input_layer  = Dense(units=200,activation="sigmoid")(input_layer)
    input_layer  = MaxPooling1D(2)(input_layer)
    input_layer  = Flatten()(input_layer)
    split_point  = Dense(units=self.n_inputs*6,activation="sigmoid")(input_layer)
    


    # for oyap timebegin
    _ = Dense(units=165, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    _ = Dense(units=155, activation='sigmoid')(_)
    # _ = Dense(units=105, activation='relu')(_)
    _ = Dense(units=65, activation='sigmoid')(_)
    # _ = Dropout(0.2)(_)
    # _ = Dense(units=65, activation='tanh')(_)
    # _ = LSTM(65)(_)
    _ = Dense(units=15, activation='sigmoid')(_)
    _ = Dense(units=10, activation='sigmoid')(_)
    timebeg_output = Dense(units=1, activation='sigmoid', name='timebeg_output')(_)
     
    # for timeval oyap prediction
    _ = Dense(units=165, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    _ = Dense(units=155, activation='sigmoid')(_)
    # _ = Dense(units=105, activation='relu')(_)
    _ = Dense(units=65, activation='sigmoid')(_)
    # _ = Dropout(0.2)(_)
    # _ = Dense(units=65, activation='tanh')(_)
    # _ = LSTM(65)(_)
    _ = Dense(units=15, activation='sigmoid')(_)
    _ = Dense(units=10, activation='sigmoid')(_)
    timeval_output = Dense(units=1, activation='sigmoid', name='timeval_output')(_)
    
    # for oyap exitions
    _ = Dense(units=253, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    # _ = Dense(units=32, activation='relu')(_)
    # _ = Dense(units=253, activation='sigmoid')(_)
    _ = Dense(units=153, activation='sigmoid')(_)
    _ = Dropout(0.2)(_)
    _ = Dense(units=53, activation='sigmoid')(_)
    # _ = Dense(units=23, activation='sigmoid')(_)
    bool_output = Dense(units=2, activation='softmax', name='bool_output')(_)
     
     
    self.set_model( Model(inputs=a, outputs=[bool_output, timebeg_output, timeval_output] ) )
    # self.set_model( Model( inputs=a, outputs=[bool_output, timebeg_output] ) )

    # 
    ## Compile model 
    # 
    # opt = SGD(lr=0.001,momentum=0.9)

    # 
    # если задаем веса
    # 
    if weights is not None:
      self._model.set_weights(weights)

    self._model.compile(
                  optimizer    = 'rmsprop',
                  # optimizer    = 'adam',
                  # optimizer    = opt,
                  # loss         = {'bool_output': 'binary_crossentropy', 
                  loss         = {'bool_output': 'categorical_crossentropy', 
                                  'timebeg_output': 'mse', 
                                  'timeval_output': 'mse'
                                  # 'timeval_output': 'mean_squared_error'
                                  },
                  # loss_weights = {'bool_output': 1.0, 
                  #                 'timebeg_output': 1.0, 
                  #                 'timeval_output': 1.0
                  #                 },
                  # metrics      = {'bool_output': 'binary_accuracy', 
                  metrics      = {'bool_output': 'categorical_accuracy', 
                                  'timebeg_output': 'accuracy', 
                                  'timeval_output': 'accuracy'
                                  }
                  )
    print(self._model.summary())

    # save model graph
    self.plot_model_fit()
    return self


  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model_worked(self,weights=None):
    ## Define the network
    # 
    a            = Input( batch_shape=(self._batch_size, self.n_backtime, self.X_train.shape[2]) )
    # b            = Input( batch_shape=(self.n_inputs*3) )
    # print(  )
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = True,
                        stateful          = True,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
                        # units=(self._batch_size, self.n_backtime, self.X_train.shape[2]),
                        # units=self.X_train.shape[2],
                       # return_sequences  = False,
                       # stateful          = False )(a)
    # input_layer2 = LSTM((self.n_inputs*3), 
    #                    stateful=False,
    #                    return_sequences=False )(input_layer)
    # _  = Dense(units=self.n_inputs*5,activation="sigmoid")(input_layer)
    # _  = Dense(units=self.n_inputs*3,activation="sigmoid")(_)
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = True,
                        stateful          = True,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(input_layer)
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = False,
                        stateful          = True,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(input_layer)
    # input_layer  = Dense(units=200,activation="sigmoid")(input_layer)
    split_point  = Dense(units=self.n_inputs*3,activation="relu")(input_layer)
    


    # for oyap timebegin
    _ = Dense(units=185, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    _ = Dense(units=155, activation='sigmoid')(_)
    # _ = Dense(units=105, activation='relu')(_)
    _ = Dense(units=55, activation='sigmoid')(_)
    _ = Dense(units=10, activation='sigmoid')(_)
    timebeg_output = Dense(units=1, activation='sigmoid', name='timebeg_output')(_)
     
    # for timeval oyap prediction
    _ = Dense(units=165, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    # _ = Dense(units=20, activation='relu')(_)
    # _ = Dense(units=255, activation='relu')(_)
    _ = Dense(units=155, activation='sigmoid')(_)
    _ = Dense(units=65, activation='sigmoid')(_)
    _ = Dense(units=15, activation='sigmoid')(_)
    timeval_output = Dense(units=1, activation='sigmoid', name='timeval_output')(_)
    
    # for oyap exitions
    _ = Dense(units=253, activation='sigmoid')(split_point)
    _ = Dropout(0.2)(_)
    # _ = Dense(units=32, activation='relu')(_)
    # _ = Dense(units=253, activation='sigmoid')(_)
    _ = Dense(units=153, activation='sigmoid')(_)
    _ = Dropout(0.2)(_)
    _ = Dense(units=53, activation='sigmoid')(_)
    # _ = Dense(units=23, activation='sigmoid')(_)
    bool_output = Dense(units=2, activation='softmax', name='bool_output')(_)
     
     
    self.set_model( Model(inputs=a, outputs=[bool_output, timebeg_output, timeval_output] ) )
    # self.set_model( Model( inputs=a, outputs=[bool_output, timebeg_output] ) )

    # 
    ## Compile model 
    # 
    # opt = SGD(lr=0.001,momentum=0.9)

    # 
    # если задаем веса
    # 
    if weights is not None:
      self._model.set_weights(weights)

    self._model.compile(
                  # optimizer    = 'rmsprop',
                  optimizer    = 'adam',
                  # optimizer    = opt,
                  # loss         = {'bool_output': 'binary_crossentropy', 
                  loss         = {'bool_output': 'categorical_crossentropy', 
                                  'timebeg_output': 'mse', 
                                  'timeval_output': 'mse'
                                  # 'timeval_output': 'mean_squared_error'
                                  },
                  # loss_weights = {'bool_output': 1.0, 
                  #                 'timebeg_output': 1.0, 
                  #                 'timeval_output': 1.0
                  #                 },
                  # metrics      = {'bool_output': 'binary_accuracy', 
                  metrics      = {'bool_output': 'categorical_accuracy', 
                                  'timebeg_output': 'accuracy', 
                                  'timeval_output': 'accuracy'
                                  }
                  )
    print(self._model.summary())

    # save model graph
    self.plot_model_fit()
    return self


  # 
  # Обучаем модель
  # 
  def fit(self):
    tensorboard = TensorBoard(log_dir="logs/keras/"+self._stantion+"_"+dt.now().strftime("%Y%m%d-%H%M%S") )

    # print([self.y_train_1,self.y_train_2,self.y_train_3])
    # return

    self.history = self._model.fit( self.X_train, 
                                    # [self.y_train_1, self.y_train_2],
                                    [self.y_train_1,self.y_train_2,self.y_train_3], 
                                    epochs          = self.num_epochs, 
                                    batch_size      = self._batch_size,
                                    validation_data = (self.X_test, [self.y_test_1,self.y_test_2,self.y_test_3] ),
                                    # validation_data = (self.X_test, [self.y_test_1,self.y_test_2 ] ),
                                    verbose         = 2, 
                                    shuffle         = False,
                                    callbacks       = [tensorboard] )

    return self


  # 
  # Рисуем графики и показываем их
  # 
  def plot(self,show=True,param="T",save=False):
    plt.style.use( self._plotstype )
    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # print(self.history.history)
    # loop over the loss names
    # for (i, l) in enumerate([ 'bool_output', 'timebeg_output' ]):
    for (i, l) in enumerate(['bool_output', 'timebeg_output', 'timeval_output']):
      # plot the loss for both the training and validation data
      title = "Loss for {}".format(l) if l != "loss" else "Total loss"
      ax[i].set_title(title)
      ax[i].set_xlabel("Epoch #")
      ax[i].set_ylabel("Loss")
      ax[i].plot(np.arange(0, self.num_epochs), self.history.history[l+"_loss"], label=l)
      # ax[i].plot(np.arange(0, self.num_epochs), self.history.history["val_" + l + "_acc"],label="val_" + l)
      ax[i].legend()

    # 
    # если в настройках установлено что показываем график
    # и мы вызываем метод не из кода
    # то показываем график
    # 
    if save:
      path = self.get_path()
      self.try_mkdir()
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_train.png")
    # if self._plot and show:
    #   plt.show()

    # return self
    
    # print(  self._scalerY.inverse_transform( 
    #           np.transpose(np.reshape(self._model.predict( self.prepare_x_to_predict(self.chunk(self.X_test,self._batch_size)[0],self.X_test.shape[2]) ) , ( self.y_train.shape[1] ,-1) ) )
    #           )
    #      )
    predicted = self._model.predict( self.X_test )
    print(predicted)
    # _shape = np.shape(predicted)
    # predicted = np.transpose( ( np.reshape(predicted,(_shape[0],_shape[1]) ) )  )
    # predicted = np.reshape(predicted,(_shape[0],_shape[1]) )

    # print(np.reshape(predicted[2],(1,-1)))
    # print(np.reshape(predicted[1],(1,-1)))
    # print((predicted[0]))

    # print( np.shape(np.reshape(predicted[2],(1,-1))) )
    # print( np.shape(np.reshape(predicted[1],(1,-1))) )
    # print( np.shape((predicted[0])) )

    predicted[2] = self._scalerY[2].inverse_transform( np.reshape(predicted[2],(1,-1)) )
    predicted[1] = self._scalerY[1].inverse_transform( np.reshape(predicted[1],(1,-1)) )
    # predicted[0] = self._scalerY[0].inverse_transform( (predicted[0]) )
    predicted = np.transpose( np.flip( predicted )  )
    
    # predicted = self._scalerY.inverse_transform( predicted )

    # ax2.plot( predicted, label="Predicted " )
    # ax2.plot( self._scalerY.inverse_transform(self.y_test), label="Real value of "+param )
    # ax2.set(title=param+' values '+(self._from_date.strftime("%Y.%m.%d"))+"-"+(self._to_date.strftime("%Y.%m.%d")), ylabel=param )
    # ax2.legend()

    # errorvalues = predicted-self._scalerY.inverse_transform( np.array(self.chunk(self.y_test,self._batch_size)).flatten() ).flatten()
    # sko         = np.std(errorvalues, ddof=1)
    # median      = np.median(errorvalues)
    # print( "SKO",sko )
    # print( "Median",median )
    # ax3.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    # ax3.hist( errorvalues , 
    #            label="loss" )
    # ax2.legend()


    # # 
    # # если в настройках установлено что показываем график
    # # и мы вызываем метод не из кода
    # # то показываем график
    # # 
    # if self._plot and show:
    #   plt.show()
    # if save:
    #   path = self._model_name+"_"+self._stantion
    #   self.try_mkdir()
    #   fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_train.png")
    return self


  # 
  # Рисуем график для верификации прогнозов
  # 
  def plot_predict(self,show=True,param="T",save=False):
    plt.style.use( self._plotstype )
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    predicted = self._model.predict( self.X_train )
    print(self.X_train)

    predicted[2] = self._scalerY[2].inverse_transform( np.reshape(predicted[2],(1,-1)) )
    predicted[1] = self._scalerY[1].inverse_transform( np.reshape(predicted[1],(1,-1)) )

    # ax1.plot( predicted[1][0], label="Predicted начало грозы " )
    _train = self._scalerY[1].inverse_transform(self.y_train_2)
    # ax1.plot( _train, label="Real value of начало грозы" )
    # ax1.set( title  = 'по данным, которых не было в обучающей выборке'+(self._from_date_predicted.strftime("%Y.%m.%d"))+"-"+(self._to_date_predicted.strftime("%Y.%m.%d")),
    #          ylabel = param )
    # ax1.legend()


    errorvalues = predicted[1][0]-_train.flatten()
    # print(predicted[1][0])
    # print(_train.flatten())
    sko         = np.std(errorvalues, ddof=1)
    median      = np.median(errorvalues)
    ax2.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    ax2.plot(  errorvalues, label="loss" )
    ax2.legend()

    ax3.plot( predicted[2][0], label="Predicted конец тумана" )
    _train = self._scalerY[2].inverse_transform(self.y_train_3)
    ax3.plot( _train, label="Real value of конец тумана" )
    ax3.legend()

    errorvalues = predicted[2][0]-_train.flatten()
    sko         = np.std(errorvalues, ddof=1)
    median      = np.median(errorvalues)
    print( "SKO",sko )
    print( "Median",median )
    ax4.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    ax4.plot(  errorvalues, label="loss" )
    ax4.legend()


    # выводим графики
    # if self._plot and show:
    #   plt.show()
    if save:
      path = self.get_path()
      self.try_mkdir()
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_predict.png")
      dfp = pd.DataFrame(np.concatenate( 
                          ( (predicted[0]) ,
                            np.reshape(predicted[1], (-1,1)) ,
                            np.reshape(predicted[2], (-1,1)) )
                          , axis=1 ) )

      dfp.columns = [ 'p_yes','p_no', 'p_val','p_beg' ]
      # dfp.loc[ dfp.p_yes >= 0.5 ,'p_yes'] = 1
      # dfp.loc[ dfp.p_yes < 0.5 ,'p_yes']  = 0
      # dfp.loc[ dfp.p_no >= 0.5 ,'p_no'] = 1
      # dfp.loc[ dfp.p_no < 0.5 ,'p_no']  = 0

      dfp.to_csv( path+"/predicted_val.csv",sep=";" )
      
      
      df = pd.DataFrame( np.concatenate( 
                          ( np.array((self.y_train_1 ))  ,
                            self._scalerY[1].inverse_transform( np.array((self.y_train_2 )) ) ,
                            self._scalerY[2].inverse_transform( np.array((self.y_train_3 )) ) )
                          , axis=1 ) 
                       )
      df.columns = [ 'yes','no', 'val','beg' ]
      df.to_csv( path+"/real_val.csv",sep=";" )

      m = pd.concat([df, dfp], axis=1, sort=False)
      m['opravd']    = m.apply( lambda row: 1 if row['yes']==1 and row['p_yes']==1 else 0 ,axis=1 )
      m['opravd_no'] = m.apply( lambda row: 1 if row['yes']==0 and row['p_yes']==0 else 0 ,axis=1 )
      m['diff_val']  = m.apply( lambda row: row['val']-row['p_val'] ,axis=1 )
      m['diff_beg']  = m.apply( lambda row: row['beg']-row['p_beg'] ,axis=1 )

      _file_name = self.get_path()+'/data_surf_w_aero_pred.csv'
      z          = pd.read_csv(_file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)
      z          = z.loc[~z.index.duplicated(keep="first")]

      # all rows
      rows = len(m.index)
      # sum of opravd
      opravd    = m['opravd'].sum()
      opravd_no = m['opravd_no'].sum()

      # 
      p_yes = m['p_yes'].sum() 
      p_no  = m['p_no'].sum()

      yes = m['yes'].sum()
      no  = m['no'].sum()

      m['Name'] = np.nan
      m['perc_opravd'] = np.nan

      # оправдываемость прогнозов
      U = (opravd+opravd_no)/rows * 100
      m['Name'].iloc[0] = 'U'
      m['perc_opravd'].iloc[0] = U

      m['Name'].iloc[1] = 'Uwith'
      m['perc_opravd'].iloc[1] = opravd/p_yes * 100

      m['Name'].iloc[2] = 'Uwout'
      m['perc_opravd'].iloc[2] = opravd_no/p_no * 100


      # 
      # =====
      # 


      m['Name'].iloc[3] = 'F count all'
      m['perc_opravd'].iloc[3] = rows

      m['Name'].iloc[4] = 'F count oyap'
      m['perc_opravd'].iloc[4] = yes

      m['Name'].iloc[5] = 'F count no oyap'
      m['perc_opravd'].iloc[5] = no

      m['Name'].iloc[6] = 'Preduprezd (yes)'
      m['perc_opravd'].iloc[6] = opravd/yes * 100

      m['Name'].iloc[7] = 'Preduprezd (no)'
      m['perc_opravd'].iloc[7] = opravd_no/no * 100

      # 
      # считаем среднее отклонение значений времни начала и продолжительности ОЯП
      # 
      df_opravd = m.loc[ m.opravd==1 ]
      df_opravd_len = len(df_opravd.index)
      
      m['Name'].iloc[9] = 'Otkl val_beg'
      m['perc_opravd'].iloc[9] = df_opravd['diff_beg'].sum()/df_opravd_len

      m['Name'].iloc[10] = 'Otkl val'
      m['perc_opravd'].iloc[10] = df_opravd['diff_val'].sum()/df_opravd_len

      # 
      # рисуем и сохраняем матрицу корреляций
      # 
      self.plot_corrmatrix()
      

      m.to_csv( path+"/real_val_and_pred.csv",sep=";" )


    return self

  
  # 
  # Рисуем матрицу корреляции на основе данных, которые пришли
  # 
  def plot_corrmatrix(self):

    file_name = self.get_path()+'/data_surf_w_aero_fact.csv'

    # 
    # загружаем все в панду
    # 
    df = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)

    df = df.drop(['Unnamed: 0','H','minute','N','NN','RR','lat','lon','stantion'],axis=1)
    df['RAIN']    = df.apply( lambda row: 1 if row['WW']==12 else 0 ,axis=1 )


    # corr = df.corr().astype(float)
    corr = df[ (df['WW']>0) ].corr().astype(float)
    # print(corr)

    # plt.fig(figsize=(40,40)) 
    import seaborn as sns
    # fig = plt.figure()
    fig,ax = plt.subplots(figsize=(17,15))
    sns.heatmap(ax=ax,data=corr, annot=True, fmt='2.2f', cmap='Blues')
    plt.title( "Матрица корреляций. Станция  "+self._stantion )
    if self._save:
      plt.savefig( self.get_path()+"/corr1_"+self._model_name+"_"+str(self._stantion)+".png")
      dd = pd.DataFrame(corr)
      dd.to_csv( self.get_path()+"/corr.csv", sep=';' )

    return self