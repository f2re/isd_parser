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

from   datetime import datetime as dt
from   datetime import date, timedelta
import time
from calendar import timegm

from .stantionclass import Stantion

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pprint import pprint

import multiprocessing, logging
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

from geopy.distance import geodesic

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout, LSTM
from keras.optimizers import SGD
from keras.models import Model

from keras.callbacks import TensorBoard

class ForecastSeparateLSTMTrainerClass(LSTMTrainerClass):

  def __init__(self):
    super(ForecastSeparateLSTMTrainerClass, self).__init__()

    return

  def train(self):
    for param in self._params:
      self.mkbatch( param ) \
         .split_batch() \
         .scale(True) 

      # 
      # train first model
      # 
      if self._model is None:
         self.set_default_model()

      self.fit() \
         .save( 'model0',md=self._model ) \
         .plot(param=param) 
         # .predict(param=param)

      self.set_next_model() \
           .fit(md=self._model_1,yt=self.y_train_2,yv=self.y_test_2) \
           .save( 'model1',md=self._model_1 ) \
           .plot(param=param,md=self._model_1) 
           # .plot_predict(param=param,save=self._save,md=self._model_1,name="1",scaler=self._scalerY[1])

      self.set_next2_model() \
           .fit(md=self._model_2,yt=self.y_train_3,yv=self.y_test_3) \
           .save( 'model2',md=self._model_2 ) \
           .plot(param=param,md=self._model_2) 
           # .plot_predict(param=param,save=self._save,md=self._model_2,name="2",scaler=self._scalerY[2])
    return self

  # 
  # loading model first
  # 
  def load_model_1(self,param):
    self.load_model(param,mdname='_model_1')
    return self
  # 
  # loading model first
  # 
  def load_model_2(self,param):
    self.load_model(param,mdname='_model_2')
    return self
  # 
  # loading model first
  # 
  def load_model_0(self,param):
    self.load_model(param,mdname='_model')
    return self

  # 
  # predict 1 model
  # 
  def plot_predict_0(self,param):
    self.plot_predict(param=param,save=self._save,md=self._model,name="0",scaler=self._scalerY[0])
    return self 

  # 
  # predict 1 model
  # 
  def plot_predict_1(self,param):
    self.plot_predict(param=param,save=self._save,md=self._model_1,name="1",scaler=self._scalerY[1])
    return self

  # 
  # predict 2 model
  # 
  def plot_predict_2(self,param):
    self.plot_predict(param=param,save=self._save,md=self._model_2,name="2",scaler=self._scalerY[2])
    return self

  # 
  # Делаем батчи
  # @param - параметр по которому будем прогнозировать (Y)
  # @wlist - список IWeatherList
  # 
  def makebatch(self,param,wlist, fromdt, todt):
    _X_batch = []
    _Y_batch = []

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
        i = 0
        x_arr = []
        for item in arraypool[0]:
          _X = item
          _X.onlyFloatOn()
          _X.calcSun()
          # print(_X.get_sunAll()['sunrise'])
          # sunrises.append( item.get_sunAll()['sunrise'] )
          # sunrises.append( item.get_sunAll()['sunset'] )
          if _X is not None and _X is not False:
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
                            _X.get_hour(), 
                            _X.get_sunAll()['sunrise'],
                            _X.get_sunAll()['sunset'],
                            _X.get_date().day,
                            _X.get_date().month ] )
          i+=1

        _X_batch.append( x_arr )
        _Y_batch.append( _Y )
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

    pool = ThreadPool(processes=self._processes)

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
    items = self._wlist_cur.get_items_by_oyap(_dt,self.n_backtime,self.n_backtime_delta,self.n_forecastdelta)

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

    # print(self.X_train)

    # self._scalerX = self._scalerX.fit(self.X_train)
    # self.X_train  = self._scalerX.transform(self.X_train)
    # self.X_test   = self._scalerX.transform(self.X_test)

    # print(np.reshape(self.X_train, ( shp[0]*shp[1], shp[2]) ))

    if fit:
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

    

    # pprint(self.y_train)
    # print(np.shape(self.y_train))
    # print("shape train" ,np.shape(self.y_train) )
    # self.y_train  = self._scalerY.transform(self.y_train)
    # self.y_test   = self._scalerY.transform(self.y_test)

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
      self._scalerY[0] = self._scalerY[0].fit((self.y_train_1) )
      self._scalerY[1] = self._scalerY[1].fit(np.reshape(self.y_train_2, (-1,1) ) )
      self._scalerY[2] = self._scalerY[2].fit(np.reshape(self.y_train_3, (-1,1) ) )

    print( 'train before transform:',((self.y_train_1 )) )
    # print( np.shape((self.y_train_1 )) )
    # self.y_train_1 = self._scalerY[0].transform(np.reshape(self.y_train_1, (-1,1) ))
    self.y_train_1 = self._scalerY[0].transform((self.y_train_1))
    self.y_train_2 = self._scalerY[1].transform(np.reshape(self.y_train_2, (-1,1) ))
    self.y_train_3 = self._scalerY[2].transform(np.reshape(self.y_train_3, (-1,1) ))

    print('train after transform: ',self.y_train_1)

    self.y_test_1 = []
    self.y_test_2 = []
    self.y_test_3 = []
    for _ in self.y_test:
      self.y_test_1.append( _[0] )
      self.y_test_2.append( _[1] )
      self.y_test_3.append( _[2] )

    # self.y_test_1 = self._scalerY[0].transform(np.reshape(self.y_test_1, (-1,1) ))
    self.y_test_1 = self._scalerY[0].transform((self.y_test_1))
    self.y_test_2 = self._scalerY[1].transform(np.reshape(self.y_test_2, (-1,1) ))
    self.y_test_3 = self._scalerY[2].transform(np.reshape(self.y_test_3, (-1,1) ))

    # print('test',self.y_test_1)

    return self


  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self):
    ## Define the network
    # 
    a            = Input( batch_shape=(self._batch_size, self.n_backtime, self.X_train.shape[2]) )
    # b            = Input( batch_shape=(self.n_inputs*3) )
    # print(  )
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = False,
                        stateful          = False,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
                        # units=(self._batch_size, self.n_backtime, self.X_train.shape[2]),
                        # units=self.X_train.shape[2],
                       # return_sequences  = False,
                       # stateful          = False )(a)
    # input_layer2 = LSTM((self.n_inputs*3), 
    #                    stateful=False,
    #                    return_sequences=False )(input_layer)
    split_point  = Dense(units=self.n_inputs*5,activation="relu")(input_layer)
    


    # for oyap timebegin
    _ = Dense(units=42, activation='sigmoid')(split_point)
    # _ = Activation("relu")(_)
    # _ = BatchNormalization()(_)
    # _ = Dropout(0.1)(_)
    # _ = Dense(units=35, activation='sigmoid')(_)
    _ = Dense(units=15, activation='sigmoid')(_)
    # _ = Dense(units=20, activation='sigmoid')(_)
    bool_output = Dense(units=1, activation='sigmoid', name='bool_output')(_)
     
     
    self.set_model( Model(inputs=a, outputs=bool_output ) )
    # self.set_model( Model(inputs=a, outputs=[bool_output, timebeg_output, timeval_output] ) )

    # 
    ## Compile model 
    # 
    opt = SGD(lr=0.01,momentum=0.9)
    self._model.compile(
                  # optimizer    = 'adam',
                  optimizer    = opt,
                  loss         = 'binary_crossentropy', 
                                  # 'timebeg_output': 'mae', 
                                  # 'timeval_output': 'mae'
                                  # 'timeval_output': 'mean_squared_error'
                                  # },
                  # loss_weights = {'bool_output': 1.0, 
                  #                 'timebeg_output': 1.0, 
                  #                 'timeval_output': 1.0
                  #                 },
                  metrics      = ['binary_accuracy'], 
                                  # 'timebeg_output': 'accuracy', 
                                  # 'timeval_output': 'accuracy'
                                  # }
                  )
    print(self._model.summary())

    # save model graph
    self.plot_model_fit()
    return self


  def set_next_model(self):
    a            = Input( batch_shape=(self._batch_size, self.n_backtime, self.X_train.shape[2]) )
    # b            = Input( batch_shape=(self.n_inputs*3) )
    # print(  )
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = False,
                        stateful          = False,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
    split_point  = Dense(units=self.n_inputs*3,activation="relu")(input_layer)
    
    # for oyap timebegin
    # _ = Dense(units=80, activation='sigmoid')(split_point)
    _ = Dense(units=30, activation='sigmoid')(split_point)
    # _ = Dropout(0.1)(_)
    _ = Dense(units=8, activation='sigmoid')(_)
    beg_output = Dense(units=1, activation='sigmoid', name='beg_output')(_)
     
     
    self._model_1 = Model(inputs=a, outputs=beg_output )

    # 
    ## Compile model 
    # 
    opt = SGD(lr=0.005,momentum=0.09)
    self._model_1.compile(
                  optimizer    = 'adam',
                  # optimizer    = opt,
                  loss         = 'mae',
                  metrics      = ['mae'] #,'mse','mape'
                  )
    print(self._model_1.summary())

    # save model graph
    self.plot_model_fit(self._model_1,'_1')
    return self

  def set_next2_model(self):
    ## Define the network
    # 
    a            = Input( batch_shape=(self._batch_size, self.n_backtime, self.X_train.shape[2]) )
    # b            = Input( batch_shape=(self.n_inputs*3) )
    # print(  )
    input_layer  = LSTM(
                        units=self.X_train.shape[2], 
                        input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                        return_sequences  = False,
                        stateful          = False,
                        batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) )(a)
    split_point  = Dense(units=self.n_inputs*5,activation="relu")(input_layer)
    # for oyap timebegin
    # _ = Dense(units=82, activation='relu')(split_point)
    _ = Dense(units=12, activation='relu')(split_point)
    # _ = Dropout(0.1)(_)
    # _ = Dense(units=52, activation='sigmoid')(_)
    _ = Dense(units=5, activation='relu')(_)
    beg_output = Dense(units=1, activation='relu', name='beg_output')(_)
     
     
    self._model_2 = Model(inputs=a, outputs=beg_output )

    # 
    ## Compile model 
    # 
    opt = SGD(lr=0.005,momentum=0.09)
    self._model_2.compile(
                  optimizer    = 'adam',
                  # optimizer    = opt,
                  loss         = 'mae',
                  metrics      = ['mae'] #,'mse','mape'
                  )
    print(self._model_2.summary())

    # save model graph
    self.plot_model_fit(self._model_2,'_2')
    return self

  # 
  # Обучаем модель
  # 
  def fit(self,md=None,yt=None,yv=None):
    tensorboard = TensorBoard(log_dir="logs/keras/"+dt.now().strftime("%Y%m%d-%H%M%S") )

    # print([self.y_train_1,self.y_train_2,self.y_train_3])
    # return
    if md is None:
      md = self._model
    if yt is None:
      yt = self.y_train_1
    if yv is None:
      yv = self.y_test_1

    self.history = md.fit( self.X_train, 
                                    # [self.y_train_1, self.y_train_2],
                                    yt, 
                                    # [self.y_train_1,self.y_train_2,self.y_train_3], 
                                    epochs          = self.num_epochs, 
                                    batch_size      = self._batch_size,
                                    validation_data = (self.X_test, yv),
                                    # validation_data = (self.X_test, [self.y_test_1,self.y_test_2,self.y_test_3] ),
                                    # validation_data = (self.X_test, [self.y_test_1,self.y_test_2 ] ),
                                    verbose         = 2, 
                                    shuffle         = False,
                                    callbacks       = [tensorboard] )

    return self


  # 
  # Рисуем графики и показываем их
  # 
  def plot(self,show=True,param="T",save=False,md=None):
    plt.style.use( self._plotstype )
    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # print(self.history.history)
    # loop over the loss names
    # for (i, l) in enumerate([ 'bool_output', 'timebeg_output' ]):
    # for (i, l) in enumerate(['bool_output', 'timebeg_output', 'timeval_output']):
    #   # plot the loss for both the training and validation data
    #   title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    #   ax[i].set_title(title)
    #   ax[i].set_xlabel("Epoch #")
    #   ax[i].set_ylabel("Loss")
    #   ax[i].plot(np.arange(0, self.num_epochs), self.history.history[l+"_loss"], label=l)
    #   # ax[i].plot(np.arange(0, self.num_epochs), self.history.history["val_" + l + "_acc"],label="val_" + l)
    #   ax[i].legend()

    # 
    # если в настройках установлено что показываем график
    # и мы вызываем метод не из кода
    # то показываем график
    # 
    # if save:
    #   path = self._model_name+"_"+self._stantion
    #   self.try_mkdir()
    #   fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_train.png")
    # if self._plot and show:
    #   plt.show()

    # return self
    
    # print(  self._scalerY.inverse_transform( 
    #           np.transpose(np.reshape(self._model.predict( self.prepare_x_to_predict(self.chunk(self.X_test,self._batch_size)[0],self.X_test.shape[2]) ) , ( self.y_train.shape[1] ,-1) ) )
    #           )
    #      )
    
    if md is None:
      md = self._model

    predicted = md.predict( self.X_test )
    _shape = np.shape(predicted)
    # predicted = np.transpose( ( np.reshape(predicted,(_shape[0],_shape[1]) ) )  )
    # predicted = np.reshape(predicted,(_shape[0],_shape[1]) )
    # print('predicted')
    # for x in predicted:
    #   print(x)
    # print(predicted)
    # print(np.reshape(predicted[2],(1,-1)))
    # print(np.reshape(predicted[1],(1,-1)))
    # print((predicted[0]))

    # print( np.shape(np.reshape(predicted[2],(1,-1))) )
    # print( np.shape(np.reshape(predicted[1],(1,-1))) )
    # print( np.shape((predicted[0])) )

    # predicted[2] = self._scalerY[2].inverse_transform( np.reshape(predicted[2],(1,-1)) )
    # predicted[1] = self._scalerY[1].inverse_transform( np.reshape(predicted[1],(1,-1)) )
    # predicted[0] = self._scalerY[0].inverse_transform( (predicted[0]) )
    # predicted = np.transpose( np.flip( predicted )  )
    
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
  def plot_predict(self,show=True,param="T",save=False,md=None,name='',scaler=None):
    plt.style.use( self._plotstype )
    fig, (ax1, ax2) = plt.subplots(2)

    if md is None:
      md = self._model
    
    predicted = md.predict( self.X_train )
    # _shape = np.shape(predicted)
    
    # predicted = np.reshape(predicted,(_shape[0],_shape[1]) )
    # predicted[2] = self._scalerY[2].inverse_transform( np.reshape(predicted[2],(1,-1)) )
    # predicted[1] = self._scalerY[1].inverse_transform( np.reshape(predicted[1],(1,-1)) )
    # predicted[0] = self._scalerY[0].inverse_transform( np.reshape(predicted[0],(1,-1)) )
    # predicted = np.transpose( np.flip( predicted )  )
    print("========== PLOT PREDICTED ===========")

    print("what is model? ",md)
    print("Name: ",name)
    print("what we predicted:",predicted)
    print("Scaler: ",scaler)
    if scaler is not None:
      predicted = scaler.inverse_transform( predicted )
      print('transformed!')
      pprint(predicted)
    # pprint(predicted)
    # print(np.shape(predicted))
    # predicted = self._scalerY.inverse_transform( predicted ) 
    # pprint(predicted)
    # pprint(self._scalerY.inverse_transform(self.y_train))
    
    # sns.catplot( x="number",hue="smoker", col="", kind="swarm", data=predicted )

    # ax1.plot( predicted, label="Predicted "+param )
    # ax1.plot( self._scalerY.inverse_transform(self.y_train), label="Real value of "+param )
    # ax1.set( title  = 'по данным, которых не было в обучающей выборке'+(self._from_date_predicted.strftime("%Y.%m.%d"))+"-"+(self._to_date_predicted.strftime("%Y.%m.%d")),
    #          ylabel = param )
    # ax1.legend()


    # errorvalues = predicted-self._scalerY.inverse_transform(np.array(self.chunk(self.y_train,self._batch_size)).flatten().reshape(-1,1) ).flatten()
    # sko         = np.std(errorvalues, ddof=1)
    # median      = np.median(errorvalues)
    # print( "SKO",sko )
    # print( "Median",median )
    # ax2.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    
    # ax2.hist(  errorvalues, label="loss" )
    # ax2.legend()
    # выводим графики
    # if self._plot and show:
      # plt.show()
    if save:
      path = self._model_name+"_"+self._stantion
      self.try_mkdir()
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_predict.png")
      df = pd.DataFrame(predicted)
      df.to_csv( path+"/predicted_val_"+name+".csv",sep=";" )
      # print( np.reshape(self._scalerY[0].inverse_transform( np.array((self.y_train_1 )) ), (-1,1) ) )
      # print(self._scalerY[1].inverse_transform( np.array((self.y_train_2 )) ))
      # print(self._scalerY[2].inverse_transform( np.array((self.y_train_3 )) ))
      df = pd.DataFrame( np.concatenate( 
                          ( np.reshape(self._scalerY[0].inverse_transform( np.array((self.y_train_1 )) ), (-1,1) ) ,
                            self._scalerY[1].inverse_transform( np.array((self.y_train_2 )) ) ,
                            self._scalerY[2].inverse_transform( np.array((self.y_train_3 )) ) )
                          , axis=1 ) 
                       )
      df.to_csv( path+"/real_val.csv",sep=";" )
    return self
