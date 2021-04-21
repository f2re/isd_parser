# -*- coding: utf-8 -*-
# 
import json
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime as dt
from  datetime import date, time, timedelta

import os.path

from collections import defaultdict
from collections import Counter

from stantionclass import Stantion
from trainer.ilistweather import IListWeather

#
# многопроцессорность
#
import multiprocessing as mp

import numpy as np
from scipy.interpolate import griddata

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from trainer.dbfunctions import *
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap
import  mpl_toolkits.basemap  

# 
# параметры подключения к монго
# 
mongo_db         = 'srcdata'
mongo_collection = 'meteoisd'
mongo_host       = 'localhost'
mongo_port       = 27017

# начало периода
dt_cur  = dt(2013, 5, 21)
dt_pred = dt(2017, 2, 1)


wlist = get_weather_by_ISDday(dt_cur, mongo_db, mongo_collection, mongo_host, mongo_port)

x,y,val = [],[],[]

for item in wlist.get_all():
  item.onlyFloatOn()
  p = item.get_P()
  if p>800 and p<1300:
    x.append(item.get_lon())
    y.append(item.get_lat())
    val.append(item.get_P())


x = [2.0, 4.0, 5.0, 8.0,16.0,20.0,  
            3.0, 4.5,10.0,16.0,20.0, 
            2.5, 6.0,14.0,16.0,20.0,    
            3.0, 6.0,10.0,14.0,16.0,20.0,    
            3.5, 5.0,12.0,16.0,20.0,    
            4.0,10.5,16.0,20.0,    
            3.0, 4.0,12.0,16.0,20.0,    
            2.0,10.0,14.0,20.0]
y = [2.0, 4.0, 5.0, 8.0,16.0,20.0,  
            2.8, 4.0, 8.8,14.0,16.5, 
            2.0, 4.8,10.8,12.1,14.0,   
            2.0, 4.0,6.2, 8.2,9.1,10.0,   
            2.0, 2.8, 5.8, 6.6, 6.9,    
            1.9, 4.0, 5.0, 4.8,    
            1.0, 1.2, 3.0, 3.5, 3.1,    
            0.2, 1.2, 1.8, 1.3]
val = [0.0, 0.0, 0.0, 0.0,0.0,0.0,  
            0.5, 0.5, 0.5, 0.5,0.5,  
            1.0, 1.0, 1.0, 1.0,1.0,
            2.0, 2.0,2.0, 2.0, 2.0,2.0,
            3.0, 3.0, 3.0, 3.0,3.0,
            5.0, 5.0, 5.0, 5.0,
           10.0,10.0,10.0,10.0,10.0,
           15.0,15.0,15.0,15.0]


val = np.array(val)
lon = np.array(x)
lat = np.array(y)

# set up plot
plt.clf()
# fig = plt.figure()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
# ax1 = fig.add_subplot(111)

# define map extent
lllon = 0
lllat = 0
urlon = 190
urlat = 80

# Set up Basemap instance
m = Basemap(ax=ax1,
    projection = 'merc',
    llcrnrlon = lllon, llcrnrlat = lllat, urcrnrlon = urlon, urcrnrlat = urlat,
    resolution='c')

m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
# grid data
numcols, numrows = 100, 100

mlon,mlat = m(x,y)

mlon = np.array(mlon)
mlat = np.array(mlat)

xi_l   = np.linspace(min(mlon), max(mlon), numcols)
yi_l   = np.linspace(min(mlat), max(mlat), numrows)
xi, yi = np.meshgrid(xi_l, yi_l)

lon_lat = np.c_[mlon.ravel(),mlat.ravel() ]


from scipy import interpolate


mmin   = (min(val)//10)*10
mmax   = max(val)

levels = np.arange(mmin,mmax,10)


# 
# Это просто заполненные поля, но неровно
# 
zi = griddata(lon_lat, val.ravel(), (xi, yi), method='linear' )
mm = ax1.contourf(xi, yi, zi, extend='both')
mc = ax1.contour(xi, yi, zi,levels, colors='white',)

# 
# ПРоинтерполированное поле
# 
# tck  = interpolate.bisplrep(mlon, mlat, val, s=28733)
# znew = interpolate.bisplev( xi_l, yi_l, tck)
# mm   = plt.contourf(xi, yi, znew, extend='both')
# mc   = plt.contour(xi, yi, znew,levels, colors='white',)

ax1.clabel(mc,inline=True,fontsize=7,fmt = '%2.1f',)
ax1.scatter(mlon, mlat, c=val, s=10,
           vmin=mmin, vmax=mmax )
ax1.scatter(xi, yi, c=zi, s=5,
           vmin=mmin, vmax=mmax )
# fig.colorbar(mm)
ax1.set_title("Pressure")
ax1.legend()


# Set up Basemap instance
m = Basemap( ax=ax2,
    projection = 'merc',
    llcrnrlon = lllon, llcrnrlat = lllat, urcrnrlon = urlon, urcrnrlat = urlat,
    resolution='c')

m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
# grid data
numcols, numrows = 100, 100

mlon,mlat = m(x,y)

mlon=np.array(mlon)
mlat=np.array(mlat)

xi_l   = np.linspace(min(mlon), max(mlon), numcols)
yi_l   = np.linspace(min(mlat), max(mlat), numrows)
xi, yi = np.meshgrid(xi_l, yi_l)

lon_lat = np.c_[mlon.ravel(),mlat.ravel() ]



from scipy.sparse import csgraph
lap                   = csgraph.laplacian(zi, normed=False)
mm1                    = ax2.contourf(xi, yi, lap, extend='both')
mc1                    = ax2.contour(xi, yi, lap, colors='white',)
ax2.set_title("Laplacian")
ax2.legend()

# 
# Рисуем поверхность 3Д
# 
# fig = plt.figure()
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)
# ax.plot_surface(xi, yi, zi,rstride=1, cstride=1, cmap=cm.jet, antialiased=True)
# ax.plot_surface(xi, yi, znew,rstride=1, cstride=1, cmap=cm.jet, antialiased=False)
plt.show()

exit()
