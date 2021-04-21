# -*- coding: utf-8 -*-

import json
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime as dt
from  datetime import date, time, timedelta

import os.path

from .stantionclass import Stantion
from .ilistweather import IListWeather
from .dbfunctions import *
from scipy import interpolate
from scipy.sparse import csgraph
from scipy.interpolate import griddata

# 
# библиотека сохранения/восстановления 
# 
import pickle

# 
# Считаем лапласиан 
# 
# 
class Laplacian(object):

  def __init__(self):
    self.mongo={
                'db':         'srcdata',
                'collection': 'meteoisd',
                'host':       'localhost',
                'port':       27017 }

    self._dt_cur  = dt(2017, 1, 1)
    self._wlist   = None
    self._results = None
    self._numcols, self._numrows = 300, 300

    self._file_prefix = "laplacian"
    self._fname       = self._file_prefix+"/"+self._file_prefix+"_"+self._dt_cur.strftime("%Y_%m_%d_%H_%M")+".dump"
    self._lap         = None
    self._zi          = None
    self._xi          = None
    self._yi          = None
    self._val         = None
    self._lon         = None
    self._lat         = None
    return

  # 
  # загружам из базы данные для расчета лапласиана
  # 
  def load(self):
    self._wlist = get_weather_by_ISDday(self._dt_cur, 
                                          db         = self.mongo['db'],
                                          collection = self.mongo['collection'],
                                          ip         = self.mongo['host'],
                                          port       = self.mongo['port'] )

    return self


  # 
  # Дулаем поле и рассчитываем лапласиан
  # 
  def calc(self):
    x,y,val = [],[],[]

    for item in self._wlist.get_all():
      item.onlyFloatOn()
      p = item.get_P()
      if p>800 and p<1300:
        x.append(item.get_lon())
        y.append(item.get_lat())
        val.append(item.get_P())
    val     = np.array(val)
    lon     = np.array(x)
    lat     = np.array(y)

    self._xi_l = np.linspace(-179, 179, self._numcols)
    self._yi_l = np.linspace(-88, 88, self._numrows)
    xi, yi     = np.meshgrid(self._xi_l, self._yi_l)

    self._lon_lat   = np.c_[lon.ravel(),lat.ravel() ]
    
    zi        = griddata(self._lon_lat, val.ravel(), (xi, yi), method='linear' )
    self._lap = csgraph.laplacian(zi, normed=False)
    
    # print(zi[56.3,45.4])

    self._zi  = zi
    self._xi  = xi
    self._yi  = yi
    self._val = val

    self._lon = lon
    self._lat = lat
    # print(self._lap)
    return self

  # 
  # загружаем из файла
  # 
  def load_fromfile(self):
    with open(self._fname, 'rb') as dump_file:
      # Step 3
      lap = pickle.load(dump_file)
      print(type(lap))
    return lap

  # 
  # сохраняем класс в файл
  # 
  def save_tofile(self):
    with open(self._fname, 'wb') as dump_file:
      # Step 3
      pickle.dump(self, dump_file)
    return self

  # 
  # если файл с дампом
  # 
  def is_fileexist(self):
    return os.path.isfile(self._fname) 

  # 
  # интерполируем данные
  # 
  def interpolate(self):
    # 
    # ПРоинтерполированное поле
    # 
    # tck         = interpolate.interp2d( self._lon, self._lat, self._val, kind='linear' )
    # tck       = interpolate.bisplrep(self._lon, self._lat, self._val, s=28733)
    tck       = interpolate.bisplrep(self._lon, self._lat, self._val, s=28733)
    self._lap = csgraph.laplacian(interpolate.bisplev( self._xi_l, self._yi_l, tck), normed=False)
    # self._lap   = csgraph.laplacian(tck( self._xi_l, self._yi_l), normed=False)
    # print(tck(56.3,5.0))

    return self


  # 
  # получаем значение по координатам
  # 
  def get_xy(self,x,y):
    zi        = griddata(np.c_[self._xi.ravel(),self._yi.ravel() ], 
                         self._lap.ravel(), (x, y), method='linear' )
    return zi

  def get_xyarr(self,arr):
    zi        = griddata(np.c_[self._xi.ravel(),self._yi.ravel() ], 
                         self._lap.ravel(), arr, method='linear' )
    return zi


  # 
  # Рисуем
  # 
  def plot(self):
    plt.contourf(self._xi, self._yi, self._lap, extend='both')
    plt.show()
    return self


  # 
  # устанавливаем дату
  # 
  def set_dt(self,val):
    self._dt_cur = val
    self._fname       = self._file_prefix+"/"+self._file_prefix+"_"+self._dt_cur.strftime("%Y_%m_%d_%H_%M")+".dump"
    return self