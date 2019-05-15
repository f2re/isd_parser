# -*- coding: utf-8 -*-
# 
import json
from pymongo import MongoClient
import numpy as np
import pandas as pd
from  datetime import datetime as dt
import psycopg2 as pg2

# подключаем библиотеку подключения к монго
import mongoconnect as mc



def read_postgres( table=None, where=None, limit=None, query=None, sort=None, offset=None ):
    """Функция запроса в БД постгрес
    
   
    Keyword Arguments:
        table {str} -- [description] (default: {'tmc_sroki'})
        where {[type]} -- [description] (default: {None})
        limit {[type]} -- [description] (default: {None})
        query {[type]} -- [description] (default: {None})
        sort {[type]} -- [description] (default: {None})
    """
    # строка подключения
    connect_str = "host='10.10.10.7' dbname='clidb_gis' user='postgres' password=''";
    # соединяемся с базой
    connect = pg2.connect(connect_str)
    # получаем курсор - указатель
    cursor = connect.cursor()

    print where

    if table is None:
        table='tmc_sroki'
    if where is None:
        where=''
    # if limit is None:
        # limit='100'
    if query is None:
        query=''
    if sort is None:
        sort=''

    cursor.execute('SELECT '+('*' if query=='' else query )+' FROM "CLIDATA".'+table+' '+( 'WHERE '+where if where!='' else '' )+' '+( 'ORDER BY '+sort if sort!='' else '' )+( '' if limit is None else ' LIMIT '+limit )+( '' if offset is None else ' OFFSET '+offset ) )
    print 'SELECT '+('*' if query=='' else query )+' FROM "CLIDATA".'+table+' '+( 'WHERE '+where if where!='' else '' )+' '+( 'ORDER BY '+sort if sort!='' else '' )+( '' if limit is None else ' LIMIT '+limit )+( '' if offset is None else ' OFFSET '+offset )

    df = pd.DataFrame(list(cursor))

    # for row in cursor.fetchall():
        # print row[3]

    return df

# 
# считываем json файл и раскодируем его
# 
def parse_json(filename = ''):
  if filename!='':

    # массив данных по строкам
    # datas=[]

    for line in open(filename,'r'):
      # загружаем строку в джсон
      js = json.loads(line)

      if js['city']['country']=="RU" :
        print json.dumps(js,indent=4)
        return

  else:
    return False
  return 




# 
# 
#  =======================
#  Парсим файлы с данными
# 

# filename = './srcdata/hourly_16.json'
# parse_json(filename)



# 
# 
# Считываем данные с постгреса
# 
# 

# итерации цикла
i=0
count=1

limit = '1000'

while count>0:

  df = read_postgres(where="TRUE",limit=limit, offset=str(i*int(limit)), table="synop96_z")
  df.columns=['station_id','date_obs','date_obs_md','tempdb','qtempdb','tempwb','qtempwb','dtempwb','tempal','qtempal','tempn','qtempn','tempx','qtempx','tempsh','qtempsh','tempdp','qtempdp','soiltmp','qsoiltmp','soiltmpa','qsoiltmpa','soiltmpn','qsoiltmpn','soiltmpx','qsoiltmpx','soiltmpsh','qsoiltmpsh','cldtot','qcldtot','cldlow','qcldlow','cldhigh','qcldhigh','cldmean','qcldmean','cldver','qcldver','cldsl','qcldsl','cldrain','qcldrain','high_low','qhigh_low','dhigh_low','cld_stn','qcld_stn','relhum','qrelhum','vappsr','qvappsr','dvappsr','vapdef','qvapdef','dvapdef','visby','qvisby','dvisby','soilstat','qsoilstat','dsoilstat','weather','qweather','ww_term','qww_term','wnddir','qwnddir','wndspdm','qwndspdm','dwndspdm','wndspdx','qwndspdx','dwndspdx','precip','qprecip','presst','qpresst','pressl','qpressl','btendch','qbtendch','btend','qbtend','qual_row','date_modif']

  count=df['station_id'].count()
  i+=1

  print "iteration "+str(i)+"..."

  if ( count==0 ):
    print "exit from script - none datas "+str(i)
    exit()

  # if ( i==3 ):
  #   exit()

  # 
  # формируем массив объектов с данными
  # 
  data_to_write = []

  for line in df.itertuples():
    # print type(line.pressl)
    # print (line.pressl)
    lobj={ 'stantion_id':line.station_id,
           'date': line.date_obs,
           'hour': line.date_obs.hour,
           'T':'' if line.tempdb is None else float(line.tempdb),
           'N': '' if line.cldtot is None else int(line.cldtot),
           'CL': '' if line.cldlow is None else int(line.cldlow),
           'CH': '' if line.cldhigh is None else int(line.cldhigh),
           'CM': '' if line.cldmean is None else int(line.cldmean),
           'rain':line.cldrain,
           'H': '' if line.high_low is None else int(line.high_low),
           'V_code':'' if line.visby is None else int(line.visby),
           'W': '' if line.weather is None else int(line.weather),
           'dd': '' if line.wnddir is None else int(line.wnddir),
           'ff': '' if line.wndspdm is None else float(line.wndspdm),
           'P': '' if line.pressl is None else float(line.pressl) }

    data_to_write.append(lobj)

  # 
  # записываем в монго
  # 
  # print data_to_write
  print mc.write_mongo( db='srcdata', collet="meteoreport", ip='localhost', port=27017, data=data_to_write )
  print 'theend'
