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

from .stantionclass import Stantion

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import multiprocessing, logging
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

from sklearn.preprocessing import MinMaxScaler

from geopy.distance import geodesic

import keras
import keras.backend as K

class RepairDataTrainerClass(LSTMTrainerClass):

  def __init__(self):
    super(RepairDataTrainerClass, self).__init__()

    # полностью станция с координатами
    self._stantion_full      = []
    # это все станции для запроса из базы
    self._stantion_list_all  = []
    # станции, по которым будем прогнозировать
    self._stantion_list      = []
    # полный список с координатами
    self._stantion_list_full = []
    # радиус поиска станций
    self._R = 2

    # 
    # Настройки многопоточности
    # 
    self.pool   = None

    # 
    # класс лапласиана, который считаем
    # 
    self._lapl = None

    # 
    # результаты предварительных расчетов используются только в makebatch
    # 
    self._results = None
    return


  # 
  # Запускаем обучение модели
  # 
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
  # Устанавливаем список станций вручную
  # 
  def set_stantions(self,val):
    self._stantion_list = val
    return self

  # 
  # Заполняем станции данными из базы (координаты названия и т.д.)
  # 
  def fill_stantions(self):
    self._stantion_list_full, self._stantion_full = get_stations(
                                date       = self._dt,
                                sts        = self._stantion_list,
                                st         = self._stantion,
                                db         = self.mongo['db'], 
                                collection = self.mongo['collection_st'], 
                                ip         = self.mongo['host'], 
                                port       = self.mongo['port'])
    self.prepare_st()
    return self

  # 
  # ищем станции в базе в радиусе поиска
  # 
  def find_stantion(self):
    self._stantion_list_full, self._stantion_full = get_stations_on_R_by_stantion(
                                date       = self._dt,
                                R          = self._R,
                                st         = self._stantion,
                                db         = self.mongo['db'], 
                                collection = self.mongo['collection_st'], 
                                ip         = self.mongo['host'], 
                                port       = self.mongo['port'])

    self.prepare_st()
    return self


  # 
  # Масштабируем наши исходные данные
  # 
  def scale(self):
    import functools
    import operator
    # 
    ## Масштабируем данные 
    # 
    # normalize the dataset
    self._scalerX  = MinMaxScaler()
    self._scalerY  = MinMaxScaler()

    shp = np.shape(self.X_train)
    # print(shp)
    # for i in self.X_train:
      # print(len(i))
    # print(np.shape( np.reshape( np.array(functools.reduce(operator.concat, self.X_train)) , ( shp[0], 107 ) ) ))

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
  # Подготавливаем станции после того как получили эти станции из базы
  # 
  def prepare_st(self):
    # crop
    if self._stantion_full in self._stantion_list_full:
      # 
      # обрезаем все, что севернее нашей станции
      # 
      lat                      = self._stantion_full['loc']['coord'][0]
      self._stantion_list_full = [s for s in self._stantion_list_full if s != self._stantion_full and s['loc']['coord'][0] < lat ]

    # 
    # сохраняем список станций, полученных из базы
    # 
    self._stantion_list_all = []
    self._stantion_list_all = [ s['st'] for s in self._stantion_list_full if s != self._stantion_full ]
    # 
    # и в конец добавляем станцию, по которой будем прогнозировать
    # 
    if (len(self._stantion_full)):
      self._stantion_list_all.append( self._stantion_full['st'] )

    return self

  # 
  # Загружаем все станции, которые есть в списке
  # 
  def load(self):

    # 
    # список именно "станций", а не классов
    # 

    if self._dbtype=="isd":
       self._weather_list, \
       self._from_date,    \
       self._to_date = get_weather_on_ISDstations(
                                date       = self._dt,
                                sts        = self._stantion_list_all,
                                offset     = 0,
                                step       = self.n_timesteps,
                                db         = self.mongo['db'], 
                                collection = self.mongo['collection'], 
                                ip         = self.mongo['host'], 
                                port       = self.mongo['port'] )
    else:
      pass
    # 
    # Сортируем по дате 
    # 
    self._weather_list.get_sort_by_time()
    self._weather_list.set_storder( self._stantion_list_all )
    self._from_cur_date = self._from_date
    self._to_cur_date   = self._to_date
    return self

    # 
  # загружаем данные для прогностических моделей
  # 
  def load_predicted(self):
    self._weather_list_predicted, \
    self._from_date_predicted,    \
    self._to_date_predicted = get_weather_on_ISDstations(
                             date       = self._dt_predicted,
                             sts        = self._stantion_list_all,
                             offset     = 0,
                             step       = self._timestep_predicted,
                             db         = self.mongo['db'], 
                             collection = self.mongo['collection'], 
                             ip         = self.mongo['host'], 
                             port       = self.mongo['port'] )
    # 
    # Сортируем по дате 
    # 
    self._weather_list_predicted.get_sort_by_time()
    self._weather_list_predicted.set_storder( self._stantion_list_all )
    self._from_cur_date = self._from_date_predicted
    self._to_cur_date   = self._to_date_predicted

    self._results       = None
    return self

  # 
  # устанавливаем модель
  # и компилируем ее сразу
  # 
  def set_default_model(self):
    ## Define the network
    # 
    print(self.X_train.shape)
    self.set_model( keras.Sequential([
        keras.layers.Dense( self.X_train.shape[1],kernel_initializer='normal', 
                            input_dim=self.X_train.shape[1], activation="sigmoid" ),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(self.n_inputs*2, activation="tanh"),
        keras.layers.Dense(self.n_inputs,  activation="relu"),
        # keras.layers.Conv2D(self.n_inputs*2,  activation="relu", kernel_size=3),
        # keras.layers.Dense(35,  activation="sigmoid"),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(20, activation="sigmoid"),
        # keras.layers.Dense(9, activation="tanh"),
        keras.layers.Dropout(0.1),
        # keras.layers.Dense(5,  activation="relu"),
        keras.layers.Dense(3,  activation="sigmoid"),
        # keras.layers.Flatten(),
        keras.layers.Dense(1, activation="tanh")
    ]) )

    # 
    ## Compile model 
    # 
    self._model.compile( 
                          # optimizer = 'adam',
                          optimizer = 'rmsprop',
                          loss      = 'mse',
                          # loss      = 'mean_absolute_error',
                          # loss      = 'mse'
                           )

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

    i        = 0
    
    if self._results is not None:
      results = self._results
    else:
      self._wlist_cur.makehash()
      results = self.get_array_mp( timedelta( hours=3 ), fromdt, todt )

    # 
    # проходимся по сплиту из 3 времен назад
    # 
    for itemarray in results:
      # 
      # забираем последнюю станцию - это та, по которой нам нужны прогнощзы
      # значения на этой стании будут использоваться дл формирования ответов
      # 
      Y_stantion = itemarray.get_all().pop()
      
      # 
      # если мы жесто хотим чтобы анализировались только 
      # параметры, когда есть правильный ответ
      # 
      if self._only_with_value:
        if Y_stantion.get_byparam(param)==0:
          continue

      # обнуляем массив
      arr = []

      startdate = None
      # 
      # Проходимяс и рассчитываем время восхода/захода
      # 
      sunrises = []
      for item in itemarray.get_all():

        if item.get_stantion()!='000000':
          if startdate is None:
            startdate = item.get_date()

          item.calcSun()
          sunrises.append( item.get_sunAll()['sunrise'] )
          sunrises.append( item.get_sunAll()['sunset'] )

      # 
      # рассчитываем лапласиан в точках (по станциям)
      # 
      # if startdate is not None:
        # 
        # пытаемся загрузить поле лапласиана из файла
        # 
        # self._lapl = self.get_laplacian_class().set_dt(startdate)
        # проверяем есть ли файл с классом
        # if self._lapl.is_fileexist():
          # если есть - загружаем
          # self._lapl = self._lapl.load_fromfile()
        # else:
          # нет - идем в базу и считаем
          # self._lapl   = self.get_laplacian_class().set_dt(startdate).load().calc()
          # self._lapl.save_tofile()

        # 
        # проходимся по станциям и считаем лапласиан
        # 
        # pool =  ThreadPool(processes=self._processes)
        # 
        # запускаем мультипроцессорные расчеты
        # 
        # laplas   = pool.map(self.get_laplacian_mp,stantionlist) 
        # pool.close()
        # print("laplas by coords x,y ", laplas)

      # 
      # это весь списко станций без последней (позже ее заберем)
      # 
      stantionlist = itemarray.get_all()[:-1]
      # print("stantion list wothout last: ",stantionlist)
      
      # формируем массив значений параметра по станциям
      # важно соблюдать порядок станций (чтобы не менялся)
      itemlist = []
      for item in stantionlist:
        itemlist = [ *itemlist, item.get_T(),item.get_P(),item.get_Td(),item.get_VV(),item.get_ff(),item.get_dd(),item.get_N(),item.get_H() ]
      
      if len(itemlist)>0 and startdate is not None:
        # 
        # здесь добавляем все остальные значения на станциях
        # 
        res = [ *itemlist, *[
                          startdate.hour, 
                          startdate.day, 
                          startdate.month ] ]

        if len(arr)==0:
          arr = res
        else:
          arr.append(res)
      i+=1
      
      if len(arr)>0:
        _X_batch.append( arr )
        # 
        # последнее значение забираем для ответа
        # 
        _Y_batch.append( Y_stantion.get_byparam(param) ) 
    # print(_X_batch)
    return _X_batch, _Y_batch
        



  # 
  # наносим станции на карту
  # 
  def plot_stantions(self):
    fig,ax = plt.subplots()
    fig.set_dpi(100)
    fig.set_figheight(13)
    fig.set_figwidth(18)
    _lat,_lon = self._stantion_full['loc']['coord']


    m = Basemap(projection='merc',llcrnrlat=_lat-4,urcrnrlat=_lat+10,\
                llcrnrlon=_lon-15,urcrnrlon=_lon+20,lat_ts=5,resolution='l',ax=ax)
    m.drawcoastlines()
    m.bluemarble(scale=1)
    # m.fillcontinents(color='white',lake_color='lightblue')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,10.))
    m.drawmeridians(np.arange(-180.,181.,10.))
    m.drawmapboundary(fill_color='lightblue')

    lon = []
    lat = []
    lon.append(_lon)
    lat.append(_lat)
    X,Y = m( lon,lat )

    ax.scatter(X,Y,color='r')
    for label, xpt, ypt in zip([self._stantion_full['st']+" ("+self._stantion_full['name']+")"], X, Y):
      plt.text(xpt+10000, ypt+5000, label,color="r")

    lon    = []
    lat    = []
    labels = []
    for item in self._stantion_list_full:
      _lat,_lon = item['loc']['coord']
      labels.append(item['st']+" ("+item['name']+")")
      lat.append(_lat)
      lon.append(_lon)
    X,Y = m(lon,lat)
    ax.scatter(X,Y,color='w')
    
    for label, xpt, ypt in zip(labels, X, Y):
      plt.text(xpt+10000, ypt+5000, label,color="w")

    # 
    # Рассчитываем дистанции до станций
    # 
    st_coords   = (self._stantion_full['loc']['coord'][0],self._stantion_full['loc']['coord'][1])
    stp_x,stp_y = m( st_coords[1],st_coords[0] )
    # 
    # Наносим эти расстояния
    # 
    for st in self._stantion_list_full:
      n_coords = (st['loc']['coord'][0],st['loc']['coord'][1])
      p_x,p_y  = m( n_coords[1],n_coords[0] )
      plt.plot( [p_x,stp_x], [p_y,stp_y] )
      plt.text( (p_x-(p_x-stp_x)/2), (p_y-(p_y-stp_y)/2), str(int(geodesic(st_coords,n_coords).km))+"km", color="w" )

    # for i, (x,y) in enumerate( zip(X,Y), start=1 ):
    #   ax.annotate(str(weather_list[i-1].get_stantion()), (x,y), xytext=(5,5),textcoords='offset points' )

    # пробуем создать папку
    self.try_mkdir()

    if self._save:
      path = self._model_name+"_"+self._stantion
      fig.savefig(path+"/map_"+self._model_name+"_"+str(self._stantion)+".png")

    plt.title("Список станций в радиусе "+str(self._R))
    plt.show()

    return self

  # 
  # Рисуем матрицу корреляции на основе данных, которые пришли
  # 
  def plot_corrmatrix(self,param):

    stantions_data = {}
    self._wlist_cur = self._weather_list

    results = self.get_array_mp( timedelta( hours=3 ),self._from_date, self._to_date )

    # 
    # проходимся по ответу и работаем с данными
    # 
    for itemarray in results:
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

    return self


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
  # Функция для многопроцессорности которая считает лапласиан
  # 
  def get_laplacian_mp(self,item):
    x,y    = item.get_lat(),item.get_lon() 
    lapval = self._lapl.get_xyarr( (x,y) )
    if lapval!=lapval or lapval is None or lapval is 0 or lapval is False:
      lapval=0.0
    return lapval

  # 
  # Получаем массив (ищем по дате)
  # 
  def get_itemarray(self,dt):
    # 
    # забираем станции за эту дату
    # 
    itemarray = self._wlist_cur.get_items_by_date_st(dt,Stantion)
    # выставляем только дробные значения (для нейросети)
    itemarray.onlyFloatOn()
    # print(dt)
    return itemarray

  # 
  # Подготавливаем данные для того, чтобы запихнуть в функцию пронозирования
  # надо ли решейпнуть массив или еще чего
  # все здесь
  # используется в функции прогнозирования
  # 
  def prepare_x_to_predict(self,x,shp):
    # print(self.X_train.shape)
    # return x.reshape(1,self.X_train.shape[1])
    # return x.reshape(1,shp)
    return x
  

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
                                     self.prepare_x_to_predict(x,self.X_test.shape)
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
                              self.prepare_x_to_predict(x,self.X_train.shape)
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
      path = self._model_name+"_"+self._stantion
      fig.savefig(path+"/figure_"+self._model_name+"_"+str(self._stantion)+"_"+param+"_predict.png")
    return self