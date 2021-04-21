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

# 
# Привер использования: 
# 
# num_epochs                = 50
# # входные параметры
# # количество входных параметров
# n_inputs                  = 11
# # шаг по времени (дни)
# n_timesteps               = 1430
# # количество измерений назад
# # еще это может быть length
# n_backtime                = 3
# # какой процент выкидываем из выборки, чтобы проверить как оно лучше работает
# n_dropout                 = 0.2

# trainer = LSTMTrainerClass()

# # 
# # Настриваем тренировочную базу
# # 
# trainer.set_stantion( '340560' )   \
#        .set_dt( dt_cur )           \
#        .set_params(['T','P','dd','ff'])             \
#        .set_timesteps( n_timesteps ) \
#        .set_inputs(n_inputs)   \
#        .set_plot(True)         \
#        .set_save(True)         \
#        .set_load(False)        \
#        .set_dbtype( 'isd' )        \
#        .set_epochs( num_epochs )        

# # 
# # Запускаем процесс
# # 
# trainer.load() \
#        .train()


import os.path
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

from   datetime import datetime as dt
from   datetime import date, timedelta
from   pymongo import MongoClient
from   collections import defaultdict
from   collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import multiprocessing, logging
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

# подключаем библиотеку подключения к монго
# import mongoconnect as mc

from .stantionclass import Stantion
from .ilistweather import IListWeather

from .dbfunctions import *
from .Laplacian import Laplacian

import keras
import keras.backend as K
from keras.utils.vis_utils import plot_model

class LSTMTrainerClass(object):

  def __init__(self):

    self.mongo={
                'db':         'srcdata',
                'collection': 'meteoisd',
                'collection_st': 'isd_stantions',
                'host':       'localhost',
                'port':       27017 }

    # текущая дата, с которой начинаем запрос в базу
    self._dt      = dt(2017, 1, 1)

    # размер ,атча
    self._batch_size = 1

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

    # Data frame of air data
    self._df_air = False
    # Dataframe of surf data
    self._df = False

    # 
    # папка и файл с данными зондирования
    # 
    self._folder   = 'TRAIN/DATA'
    self._filename = ''

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
    self.n_backtime       = 3
    self.n_backtime_delta = 3
    # время на которое прогнозируем
    self.n_forecastdelta  = 6
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
    self._stantion_name = ""

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
    # Количество процессов для многопоточного задания
    # 
    self._processes    = 8

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
    self._weather_list = IListWeather([])
    self._weather_list_predicted = IListWeather([])
    # даты запроса из базы
    self._from_date = None
    self._to_date   = None
    self._from_date_predicted = None
    self._to_date_predicted   = None

    # 
    # текущая дата, от которой будем анализировать 
    # данные из базы
    # выставляется автоматически
    # 
    self._from_cur_date = None
    self._to_cur_date   = None

    # 
    # результаты предварительных расчетов используются только в makebatch
    # 
    self._results = None

    # 
    # если надо только параметры со значениями
    # 
    self._only_with_value = False

    self._laplacian = Laplacian()
    return

  # 
  # Запускаем обучение модели
  # 
  def train(self):
    for param in self._params:
      self.mkbatch( param ) \
         .split_batch() \
         .scale() 
         # .preparesample() 

      if self._model is None:
         self.set_default_model()

      self.fit() \
         .save( param ) \
         .plot(param=param) \
         .predict(param=param)
    return self

  def load(self):
    # имя файла для сохранения/восстановления
    path = self.get_path()
    self.fillfromdt(self._dt, self.n_timesteps)
    filename       = path+'/prepared_'+self._from_date.strftime('%Y%m%d')+'_to_'+self._to_date.strftime('%Y%m%d')+'.dump'
    filename_wlist = path+'/weatherlist_'+self._from_date.strftime('%Y%m%d')+'_to_'+self._to_date.strftime('%Y%m%d')+'.dump'
    # print(self._load)
    if self._load is True:
      self.try_mkdir()
      if os.path.exists(filename_wlist):
        file = open(filename_wlist,'rb')
        self._weather_list, self._from_date, self._to_date = pickle.load(file)
        file.close()
        return self
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
    if self._dbtype=="pandas":
      self._weather_list, \
      self._from_date,    \
      self._to_date = get_weather_from_pandas(
                               date       = self._dt,
                               date_to    = self._dt_to,
                               file       = self._folder+"/"+self._filename, 
                                )

    
    if self.save:
      self.try_mkdir()
      file = open(filename_wlist,'wb')
      pickle.dump( [self._weather_list, self._from_date, self._to_date], file )
      file.close()
    return self

  # 
  # парсим данные из файла csv
  # 
  def parse_aero(self):
    if os.path.exists( self._folder+"/"+self._filename+"-air.csv" ) is False:
      filedata = parse_aero_from_csv(
                                 file       = self._folder+"/"+self._filename, 
                                  )
    return self

  # 
  # загружаем данные из файла
  # 
  def load_aero(self):
    if os.path.exists(  self._folder+"/"+self._filename+"-air.csv" ) :
      df = pd.read_csv( self._folder+"/"+self._filename+"-air.csv", sep=';', low_memory=False , index_col='date', parse_dates=True)
      self._df_air = df
    return self

  # 
  # Сливаем вместе данные зондирования и приземные данные
  # 
  def merge_air_and_surf( self, _type="fact" ):
    # 
    # Конвертируем текущую погоду в датафрейм
    # 
    file_name = self.get_path()+'/data_surf_'+_type+'.csv'
    if os.path.exists(file_name) and self._load is True:
      df_full = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)
      print("read from file "+_type+" / "+file_name)
    else:
      if ( _type=="fact" ):
        df_full = self._weather_list.onlyFloatOn().toDataFrame()
      else:
        df_full = self._weather_list_predicted.onlyFloatOn().toDataFrame()
      print("write to file "+_type+" / "+file_name)

      df_full.to_csv(file_name,sep=';')
      # this is bug with datetime and object column
      df_full = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)

    df_full = df_full.drop( ['850_T','850_D','850_dd','850_ff','925_T','925_D','925_dd','925_ff'] , axis=1 )
    # df_full.set_index(['date'])
    
    file_name = self.get_path()+'/data_surf_w_aero_'+_type+'.csv'

    if os.path.exists(file_name) and self._load is True:
      df_full = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)
    else:
      # print(df_full)
      # print(self._df_air)
      df_full = df_full.merge( self._df_air.loc[ (self._df_air.L==850.0) ].rename(columns={"T": "850_T", "D": "850_D", "dd": "850_dd", "ff": "850_ff"})[['850_T','850_D','850_dd','850_ff']],
                                on="date",
                                how="left" )

      df_full = df_full.merge( self._df_air.loc[ (self._df_air.L==925.0) ].rename(columns={"T": "925_T", "D": "925_D", "dd": "925_dd", "ff": "925_ff"})[['925_T','925_D','925_dd','925_ff']],
                                on="date",
                                how="left" )

      df_full = df_full.replace(-9999.0, np.nan).replace(-999.9, np.nan).replace(9999.9, np.nan).replace(9999.0, np.nan).replace(999.9, np.nan).replace(999.0, np.nan)

      df_full = df_full.interpolate(method='linear', axis=0).ffill().bfill()
      
      # MRi = (Tb - Tsfc) / u2
      df_full['MRi'] = (df_full['925_T']-df_full['T'])*df_full['925_ff']

      # FSI = 2 * (Ts - Tds) + 2 * (Ts - T850) + W850 
      df_full['FSI'] = 2*(df_full['T']-df_full['Td']) + 2*(df_full['T']-df_full['850_T']) + df_full['850_ff']


      df_full.to_csv(file_name,sep=';')
      # df_full = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)
    
    #  
    # конвертируем обратно из датайрейма
    # 
    weather = IListWeather([])
    weather.fromDataFrame(df_full).onlyFloatOn()
    
    if ( _type=="fact" ):
      self._weather_list = weather
    else:
      self._weather_list_predicted = weather

    return self


  # 
  # получаем правильный путь
  # 
  def get_path(self):
    return self._model_name+"_"+self._stantion+"_"+self._stantion_name

  # 
  # заполняем время начала и окончания исходя из даты и шагов
  # 
  def fillfromdt(self,dt,step):
    self._from_date = dt
    self._to_date = dt + timedelta( days=step )
    return self

  def fillfromdt_predict(self,dt,step):
    self._from_date_predicted  = dt
    self._to_date_predicted    = dt + timedelta( days=step )
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
  # пробуем заполнить имя станции
  # из базы
  # 
  def try_fill_stantion_name(self):
    if self._stantion!='':
      # получаем список станций
      stantion_list = getStantions( self.mongo['db'], 
                                    'isd_stantions', 
                                    self.mongo['host'], 
                                    self.mongo['port']  )
      if self._stantion in stantion_list.keys():
        self._stantion_name = stantion_list[self._stantion]['name']
        print(self._stantion_name)
    return self

  # 
  # преобразуем исходные данные в батчи пригодные для обучения
  # 
  def mkbatch(self,param):
    self.X_batch               = []
    self.Y_batch               = []
    self._wlist_cur            = self._weather_list
    self.X_batch, self.Y_batch = self.makebatch( param, self._weather_list, self._from_date, self._to_date )
    return self

  # 
  # 
  # 
  def mkbatch_predict(self,param):
    self.X_batch               = []
    self.Y_batch               = []
    self._wlist_cur            = self._weather_list_predicted
    self.X_batch, self.Y_batch = self.makebatch( param, self._weather_list_predicted, self._from_date_predicted, self._to_date_predicted )
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
    results = self.get_array_mp( timedelta( hours=3 ), fromdt, todt )

    # 
    # теперь надо получить набор данных с определенной заблаговременностью
    # 
    # здесь это массив IListweather
    # с шагом по времени
    # 
    for arraypool in results:
      # 
      # забираем последнюю станцию - это та, по которой нам нужны прогнощзы
      # значения на этой стании будут использоваться дл формирования ответов
      # 

      if arraypool is not False:
        _Y = arraypool.pop(0)
        # 
        # проходимся по всем данным
        # 
        i = 0
        x_arr = []
        for item in arraypool:
          _X = item
          if _X is not None and _X is not False:
            # 
            # отсутпаем назад на заданную дельту
            # 
            x_arr.append( [  _X.get_T(), 
                                    _X.get_P(), 
                                    _X.get_N(), 
                                    _X.get_dd(), 
                                    _X.get_VV(), 
                                    _X.get_ff(), 
                                    _X.get_H(),                       
                                    _X.get_RR(),                       
                                    _X.get_hour(), 
                                    _X.get_date().day,
                                    _X.get_date().month ] )
          i+=1

        _X_batch.append( x_arr )

        app = None
        if param=="T":  
          app = _Y.get_T()
        elif param=="P":
          app = _Y.get_P()
        elif param=="V":
          app = _Y.get_VV()
        elif param=="dd":
          app = _Y.get_dd()
        elif param=="ff":
          app = _Y.get_ff()

        _Y_batch.append( app )
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

    shp = np.shape(self.X_train)


    # self._scalerX = self._scalerX.fit(self.X_train)
    # self.X_train  = self._scalerX.transform(self.X_train)
    # self.X_test   = self._scalerX.transform(self.X_test)

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


    x_sz = (self.X_train.shape[0]//self._batch_size)*self._batch_size
    self.X_train = self.X_train[0:x_sz]
    self.y_train = self.y_train[0:x_sz]

    x_sz = (self.X_test.shape[0]//self._batch_size)*self._batch_size
    self.X_test = self.X_test[0:x_sz]
    self.y_test = self.y_test[0:x_sz]

    return self

  # 
  # рисуем граф модели
  # 
  def plot_model_fit(self,model=None,name=''):
    # save model figure
    path = self.get_path()
    self.try_mkdir()

    print_model = model
    if print_model is None:
      print_model=self._model

    plot_model(print_model, to_file=path+'/model'+name+'_plot.png', show_shapes=True, show_layer_names=True)
    return

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
  def prepare_XY_sample(self, dataX, dataY, length):
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
  # Рисуем матрицу корреляции на основе данных, которые пришли
  # 
  def plot_corrmatrix(self):

    file_name = self.get_path()+'/data_surf_w_aero_fact.csv'

    # 
    # загружаем все в панду
    # 
    df = pd.read_csv(file_name,sep=';', low_memory=False , index_col='date', parse_dates=True)

    df = df.drop(['Unnamed: 0','H','minute','N','NN','RR','lat','lon','stantion','WW'],axis=1)
    df['fog']    = df.apply( lambda row: 1 if row['VV']<=1000 else 0 ,axis=1 )


    # corr = df.corr().astype(float)
    corr = df[ (df['VV']<3000) ].corr().astype(float)
    # print(corr)

    # plt.fig(figsize=(40,40)) 
    import seaborn as sns
    # fig = plt.figure()
    fig,ax = plt.subplots(figsize=(17,15))
    sns.heatmap(ax=ax,data=corr, annot=True, fmt='2.2f', cmap='Greens')
    plt.title( "Матрица корреляций. Станция  "+self._stantion )
    if self._save:
      plt.savefig( self.get_path()+"/corr1_"+self._model_name+"_"+str(self._stantion)+".png")
      dd = pd.DataFrame(corr)
      dd.to_csv( self.get_path()+"/corr.csv", sep=';' )

    return self

  # 
  # Сохраняем модель и результаты в папку
  # 
  def save(self,param,md=None):
    # пробуем создать папку
    self.try_mkdir()

    if self._save is True:
      # задаем имя папки куда складывать будем модели
      path = self.get_path()
      
      plt.rcParams["figure.figsize"]=(25, 10)#Размер картинок

      if md is None:
        md = self._model

      # serialize model to JSON
      model_json = md.to_json()
      with open( path+"/model_"+str(self._stantion)+"_"+param+".json", "w" ) as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      md.save_weights(path+"/model_"+str(self._stantion)+"_"+param+".h5")
      md.save(path+"/model_"+str(self._stantion)+"_"+param+"_full.h5")
      print("Saved model <"+self._model_name+"> at <"+str(self._stantion)+"> stantion to disk")

      # 
      # сохраняем маштабаторы
      # 
      joblib.dump(self._scalerX, path+'/scaler_scalerX.sav') 
      joblib.dump(self._scalerY, path+'/scaler_scalerY.sav') 
      # joblib.dump(self._scalerX, path+'/scaler_scalerX.sav') 

      # 
      # Сохраняем картинки
      # 
      # plot history
      self.plot(show=False,param=param,save=True)
    return self

  # 
  # загружаем модель
  # 
  def load_model(self,param,mdname=None):
    # задаем имя папки куда складывать будем модели
    path = self.get_path()

    # with open( path+"/model_"+str(self._stantion)+"_"+param+".json", "r" ) as json_file:
    #   # serialize model to JSON
    #   self._model = keras.models.model_from_json( json_file.read() )

    #   # serialize weights to HDF5
    #   self._model.load_weights(path+"/model_"+str(self._stantion)+"_"+param+".h5")
    #   print("Loaded model <"+self._model_name+"> at <"+str(self._stantion)+"> stantion to disk")
    if mdname is None:
      mdname = '_model'

    self.__dict__[mdname] = keras.models.load_model(path+"/model_"+str(self._stantion)+"_"+param+"_full.h5")
    print("Loaded model <"+self._model_name+"> at <"+str(self._stantion)+"> stantion from disk")
    # print(self.__dict__[mdname])

    # 
    # load scalers
    # 
    self._scalerX = joblib.load(path+'/scaler_scalerX.sav') 
    self._scalerY = joblib.load(path+'/scaler_scalerY.sav') 
    # print( self._scalerX.data_max_, self._scalerX.data_min_ )
    # print( self._scalerY[0].data_max_, self._scalerY[0].data_min_ )
    # print( self._scalerY[1].data_max_, self._scalerY[1].data_min_ )
    # print( self._scalerY[2].data_max_, self._scalerY[2].data_min_ )
    return self

  # 
  # создаем папку с моделью, если не создано еще
  # 
  def try_mkdir(self):
    if self._save is True:
      # задаем имя папки куда складывать будем модели
      path = self.get_path()
      # 
      # если папки нет, создаем ее
      # 
      if not os.path.exists(path):
        os.makedirs(path)
    return self

  # 
  # Подготавливаем данные для того, чтобы запихнуть в функцию пронозирования
  # надо ли решейпнуть массив или еще чего
  # все здесь
  # используется в функции прогнозирования
  # 
  def prepare_x_to_predict(self,x,shp):
    xx = x.reshape(self._batch_size,self.n_backtime,shp)
    return xx

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
                                     self.prepare_x_to_predict(x,self.X_test.shape[2])
                                    ) 
                                  ),  self.chunk(self.X_test,self._batch_size) ))).flatten()
    ax2.plot( predicted, label="Predicted " )
    ax2.plot( self._scalerY.inverse_transform(self.y_test), label="Real value of "+param )
    ax2.set(title=param+' values '+(self._from_date.strftime("%Y.%m.%d"))+"-"+(self._to_date.strftime("%Y.%m.%d")), ylabel=param )
    ax2.legend()

    errorvalues = predicted-self._scalerY.inverse_transform( np.array(self.chunk(self.y_test,self._batch_size)).flatten().reshape(-1,1) ).flatten()
    sko         = np.std(errorvalues, ddof=1)
    median      = np.median(errorvalues)
    print( "SKO",sko )
    print( "Median",median )
    ax3.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    ax3.hist( errorvalues , 
               label="loss" )
    ax2.legend()


    # 
    # если в настройках установлено что показываем график
    # и мы вызываем метод не из кода
    # то показываем график
    # 
    if self._plot and show:
      plt.show()
    if save:
      path = self.get_path()
      self.try_mkdir()
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_train.png")
    return self

  # 
  # делим на чанки массив
  # 
  def chunk(self,arr,chun):
    # looping till length arr 
    n = max(1, chun)
    ret = []
    for i in range(0, len(arr), n):
      if (len(arr[i:i+n])==chun):
        ret.append(arr[i:i+n])
    return ret
    # for i in range(0, len(arr), chun):  
    #     yield arr[i:i + chun] 

  # 
  # Рисуем график для верификации прогнозов
  # 
  def plot_predict(self,show=True,param="T",save=False):
    plt.style.use( self._plotstype )
    fig, (ax1, ax2) = plt.subplots(2)
    predicted = np.array(list(map( lambda x: self._scalerY.inverse_transform( 
                          self._model.predict( 
                              self.prepare_x_to_predict(x,self.X_train.shape[2])
                            ) 
                          ),  self.chunk(self.X_train,self._batch_size) ))).flatten()
        
    ax1.plot( predicted, label="Predicted "+param )
    ax1.plot( self._scalerY.inverse_transform(self.y_train), label="Real value of "+param )
    ax1.set( title  = 'Значения температуры по данным, которых не было в обучающей выборке'+(self._from_date_predicted.strftime("%Y.%m.%d"))+"-"+(self._to_date_predicted.strftime("%Y.%m.%d")),
             ylabel = param )
    ax1.legend()


    errorvalues = predicted-self._scalerY.inverse_transform(np.array(self.chunk(self.y_train,self._batch_size)).flatten().reshape(-1,1) ).flatten()
    sko         = np.std(errorvalues, ddof=1)
    median      = np.median(errorvalues)
    print( "SKO",sko )
    print( "Median",median )
    ax2.set(title="Гистограмма распределения абсолютных ошибок СКО="+("%2.1f" % sko)+" Медиана="+("%2.1f" % median), ylabel=param)
    
    ax2.hist(  errorvalues, label="loss" )
    ax2.legend()
    # выводим графики
    if self._plot and show:
      plt.show()
    if save:
      path = self.get_path()
      self.try_mkdir()
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_predict.png")
    return self

  # 
  # Обучаем модель
  # 
  def fit(self):
    self.history = self._model.fit( self.X_train, 
                                    self.y_train, 
                                    epochs          = self.num_epochs, 
                                    batch_size      = self._batch_size,
                                    validation_data = (self.X_test, self.y_test), 
                                    verbose         = 2, 
                                    shuffle         = False )
    return self

  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self,weights=None):
    ## Define the network
    # 
    self.set_model( keras.Sequential([
        keras.layers.LSTM(self.X_train.shape[2], 
                          input_shape       = (self._batch_size,self.n_backtime,self.X_train.shape[2]), 
                          return_sequences  = True,
                          stateful          = False,
                          batch_input_shape = (self._batch_size,self.n_backtime,self.X_train.shape[2]) ),
        keras.layers.LSTM(self.n_inputs*3, stateful=False,return_sequences=False),
        # keras.layers.Dense(20,activation="relu"),
        # keras.layers.Dense(3,activation="sigmoid"),
        keras.layers.Dense(1,activation="sigmoid")
    ]) )

    # 
    ## Compile model 
    # 
    self._model.compile(
                  optimizer = 'adam',
                  # optimizer = 'adagrad',
                  # loss    = 'mean_absolute_error',
                  loss    = 'mean_squared_error',
                  metrics = ['accuracy']
                  )
    print(self._model.summary())
    return self

  # 
  # Получаем лапласиан
  # 
  def get_laplacian_class(self):
    return self._laplacian

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
        .plot_predict(show=self._plot,param=param,save=self._save)

    return self

  # 
  # пользовательский прогноз по данным из функции
  # 
  def custom_predict(self, data=None):
    if data is None:
      data  = self.X_train
    else:
      self.X_train = data

    old_weights = self._model.get_weights()
    
    # устанавливаем размер батча для прогнозов
    self.set_batch_size(1)
    # заново создаем модель с одним батчем
    self.set_default_model(weights=old_weights)
    # self._model.set_weights(old_weights)


    predict   = self._model.predict(data,batch_size=1)
    # print( dict( zip(self._scalerY[0].classes_,self._scalerY[0].transform(self._scalerY[0].classes_) ) ) )
    result = predict[0][0]
    print(result)
    raw    = [ result[0],result[1] ]
    if result[0]>0.5:
      result[0] = 1
    else:
      result[0] = 0

    if result[1]>0.5:
      result[1] = 1
    else:
      result[1] = 0
    

    # print( self._scalerY[0].inverse_transform(predict[0]) )
    predict[1] = self._scalerY[1].inverse_transform(predict[1]) 
    predict[2] = self._scalerY[2].inverse_transform(predict[2]) 
    return predict

  # 
  # делаем поиск по списку данных по времени 
  # @delta - шаг по времени
  # 
  def get_array_mp(self, delta, fromdt, todt):

    startdate = fromdt
    
    # 
    # делаем массив дат, по окторым будем искать
    # 
    dates  = []
    while startdate <= todt:
      dates.append(startdate)
      # делаем шаг по дате
      startdate += delta


    # self.pool =  Pool(processes=6)
    pool =  ThreadPool(processes=self._processes)

    # 
    # запускаем мультипроцессорные расчеты
    # 
    results   = pool.map(self.get_itemarray,dates) 
    pool.close()
    # pool.terminate()
    # pool.join()
    self._results = results

    return results

  # 
  # Получаем массив (ищем по дате)
  # 
  def get_itemarray(self,dt):
    # 
    # забираем станции за эту дату
    # 
    # print(self._wlist_cur)
    item = self._wlist_cur.get_item_by_time_withoutst(dt,self.n_backtime,self.n_backtime_delta,self.n_forecastdelta)
    return item



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

  def set_model_name(self,val):
    self._model_name = val
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

  def set_stantion_name(self,val):
    self._stantion_name = val
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
    self._dt_predicted=val
    return self

  def set_timestep_predicted(self,val):
    self._timestep_predicted=val
    return self

  def set_shuffle(self,val):
    self._shuffle = val
    return self

  def set_batch_size(self,val):
    self._batch_size = val
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

  def set_backtimedelta(self,val):
    self.n_backtime_delta = val
    return self

  def set_backtime(self,val):
    self.n_backtime = val
    return self

  def set_forecastdelta(self,val):
    self.n_forecastdelta = val
    return self

  # задаем имя csv файла с аэрологией
  def set_filename(self,val):
    self._filename = val
    return self

  # 
  # устанавливаем количество тредов
  # 
  def set_processes(self,val):
    self._processes = val
    return self

  # 
  # 
  # 
  def set_onlyvalues(self):
    self._only_with_value = True
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
