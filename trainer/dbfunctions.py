# -*- coding: utf-8 -*-

# 
# 
# Функции для работы с базой
# 
# 

from   datetime import datetime as dt
from   datetime import date, timedelta
# подключаем библиотеку подключения к монго
from . import mongoconnect as mc
from pymongo import CursorType
import psycopg2 as pg2
import pandas as pd
from .stantionclass import Stantion
from .ilistweather import IListWeather

import re

# 
# 
#  Считываем погоду по дням
#  

def get_weather_by_day( date, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=0,minute=0,second=59 )
  # ищем станции рядом 
  query = { "dt": { "$gte":from_date , "$lte":to_date } }
  
  res      = get_query( db, collection, ip, port, query )

  # получаем список станций
  stantion_list = getStantions(db, 'isd_stantions', ip, port)


  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    st.fromMongoISDFormat(doc)

    # заполняем данные по станции
    if st.get_stantion() in stantion_list.keys():
      st.fromStantionISDFormat( stantion_list[st.get_stantion()] )

    weather.append( st )
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
# Получаем погоду по России за определенную дату
# 
def get_weather_by_ISDdayNonStandartSrok(date, db, collection, ip, port):
   # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=0,minute=59,second=59 )

  # сроки, которые исключаем
  non_dates = [ date.replace( hour=_dt,minute=0,second=0 ) for _dt in range(0,23,1) ]
  print(non_dates)
  # ищем станции рядом 
  query = { "dt": { "$gte":from_date , "$lte":to_date, "$nin":non_dates } }
  
  res      = get_query( db, collection, ip, port, query )

  # получаем список станций
  stantion_list = getStantions(db, 'isd_stantions', ip, port)


  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    st.fromMongoISDFormat(doc)

    # заполняем данные по станции
    if st.get_stantion() in stantion_list.keys():
      st.fromStantionISDFormat( stantion_list[st.get_stantion()] )

    weather.append( st )
  print(count)
  # возвращаем станции
  return weather

# 
# Получаем погоду по России за определенную дату
# 
def get_weather_by_ISDday(date, db, collection, ip, port):
  # ищем станции рядом 
  query = { "dt": date }
  res   = get_query( db, collection, ip, port, query )

  # получаем список станций
  stantion_list = getStantions(db, 'isd_stantions', ip, port)

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoISDFormat(doc)

    # заполняем данные по станции
    if item.get_stantion() in stantion_list.keys():
      item.fromStantionISDFormat( stantion_list[item.get_stantion()] )

    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list

# 
# Запрос для базы которая ISD (распарсена от НОАА)
# 
def get_weather_on_ISDstation( date, st, offset, step, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )
  print('get_weather_on_ISDstation')
  # ищем станции рядом 
  query = { "dt": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "st":st }

  res = get_query( db, collection, ip, port, query )

  # получаем список станций
  stantion_list = getStantions(db, 'isd_stantions', ip, port)

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count += 1
    item   = Stantion()
    item.fromMongoISDFormat(doc)

    # заполняем данные по станции
    if item.get_stantion() in stantion_list.keys():
      item.fromStantionISDFormat( stantion_list[item.get_stantion()] )

    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list, from_date, to_date


# 
# Запрос для базы которая ISD (распарсена от НОАА)
# запрашиваем по всем станциям в списке
# 
def get_weather_on_ISDstations(date, sts, offset, step, db, collection, ip, port):
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )
  print('get_weather_on_ISDstations')
  # ищем станции рядом 
  query = { "dt": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "st": { "$in":sts } }

  res = get_query( db, collection, ip, port, query )

  # получаем список станций
  stantion_list = getStantions(db, 'isd_stantions', ip, port)

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoISDFormat(doc)
    # заполняем данные по станции
    if item.get_stantion() in stantion_list.keys():
      item.fromStantionISDFormat( stantion_list[item.get_stantion()] )
    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list, from_date, to_date

# 
# 
# Функция чтения данных из файла pandas
# 
# 
def get_weather_from_pandas(date, date_to, file):
  df = pd.read_csv(file,sep=';', low_memory=False , index_col='dt')
  prepared = df[ date.strftime("%Y-%m-%d %H:%M:%S") : date_to.strftime("%Y-%m-%d %H:%M:%S") ]

  weather_list = IListWeather([])
  for index, row in prepared.iterrows():
    item = Stantion()
    item.fromPandasFormat(row,index)
    weather_list.add( item )
  return weather_list, date, date_to

# 
# парсим данные в csv из файла
# 
def parse_aero_from_csv( file ):
  
  fout = open(file+"-air.csv", "wt")

  count = 0
  print("\nUsing readline()") 

  curdate = False

  # create DF
  # df = pd.DataFrame(columns = [ 'dt','L','H','T','D','dd','ff' ])
  # df.columns = [ 'dt','L','H','T','D','dd','ff' ]
  fp = open(file,'r')
  lines = fp.readlines()

  fout.write( ";".join(['date','L','H','T','D','dd','ff'])+"\n" )

  for line in lines:
    line = line.strip()

    # если начинается с решетки, то меняем дату
    if line[0]=="#":
      splitted = re.sub('\s+',' ',line).split(' ')
      hour = int(splitted[4])
      hour = 0 if hour>23 else hour
      curdate = dt( int(splitted[1]), int(splitted[2]), int(splitted[3]), hour, 0 )
    else:
      # 21 -9999  99500A  149    95A  710 -9999 -9999 -9999 
      # 21 -9999 100200A  149    32A  840 -9999 -9999 -9999 
      # 10 -9999  50000  5560B -208B  380 -9999   350   240 
      data = []
      if curdate is not False:
        data.append(curdate.strftime('%Y-%m-%d %H:%M:%S')) #['dt'] 
        # print(data) 
        # уровень
        data.append(str(float(re.sub('[a-zA-Z]*','',line[8:16]).strip())/100) ) #['L']   
        # высота
        data.append(str(float(re.sub('[a-zA-Z]*','',line[17:22]).strip())) ) #['H']   
        # высота
        data.append(str(float(re.sub('[a-zA-Z]*','',line[23:28]).strip())/10) ) #['T']   
        data.append(str(float(re.sub('[a-zA-Z]*','',line[34:39]).strip())/10) ) #['D']   
        data.append(str(float(re.sub('[a-zA-Z]*','',line[40:45]).strip())) ) #['dd']  
        data.append(str(float(re.sub('[a-zA-Z]*','',line[46:51]).strip())/10) ) #['ff']  

      # df.loc[count] = data
      count += 1

      # line=re.sub('\s+',',',line)

      fout.write(";".join(data)+"\n")

  # print(df.head())
  # df.to_csv(file+"-air.csv",sep=';')

  fout.close()
  print("Aero file "+file+"-air.csv"+" saved!")

  return data

# 
# Получаем станции из базы
# 
def getStantions(db, collection, ip, port):
  query         = {}
  res           = get_query( db, collection, ip, port, query )
  stantion_list = {}

  for doc in res:
    stantion_list.update( {doc['st']: doc} )
  # 
  # return list with key { '02342':{stinfo} }
  # 
  return stantion_list



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
  return weather_list, from_date, to_date


# 
# Запрос для базы которая наша
# 
def get_weather_by_day_std( date, db, collection, ip, port ): # подключаемся к БД
  
  # ищем станции рядом 
  query = { "dt": date }
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
  print("DB QUERY: "+str(query))
  # делаем запрос в базу
  # print(CursorType.EXHAUST)
  # cursor_type=CursorType.EXHAUST,
  res          = db.find(query,  batch_size=10000)
  return res

# 
# Запрос aggregate 
# 
# 
def get_query_aggregate( db, collection, ip, port, pipeline={} ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)
  print("DB aggregate: "+str(pipeline))
  # делаем запрос в базу
  # print(CursorType.EXHAUST)
  # cursor_type=CursorType.EXHAUST,
  res          = db.aggregate(pipeline)
  return res


# 
#  Считываем станции в радиусе одной станции
#  
# @lat - # долгота
# @lon - # широта
# @date - дата за которую ищем станции
# @R - # радиус поиска станции

def get_stations_on_R(lat, lon, R, date, db, collection, ip, port  ):
  # ищем станции рядом 
  query = { "date":date, "station.location.coordinates": {"$within": {"$center": [[lat, lon], R]}} }

  res      = get_query( db, collection, ip, port, query )
  count    = 0
  stations = []
  # проходимся по ответу
  for doc in res:
    # если станция уже есть - не добавляем
    if doc['station'] not in stations:
      count+=1
      stations.append( doc['station'] )

  print( "All found records: "+str(count)+"; Stantions: "+str(len(stations)) )
  # возвращаем станции
  return stations

# 
# Ищем станции в радиусе одной станции
# 
# 
def get_stations_on_R_by_stantion(st, R, date, db, collection, ip, port  ):
  # ищем станции рядом 
  query = {"station.stantion": st, "date":date }

  res      = get_query( db, collection, ip, port, query )
  count    = 0
  stations = []
  station  = []
  # проходимся по ответу
  for doc in res:
    # если станция уже есть - не добавляем
    if count==0:
      station  = doc['station']
      lat,lon  = doc['station']['location']['coordinates']
      stations = get_stations_on_R( lat, lon ,R, date, db, collection, ip, port )

  # возвращаем станции
  return stations, station

# 
# Заполняем станции данными о них
# получаем список станций которые будут использоваться в расчетах
# и станцию по которой будет составлятсья пронроз
# 
def get_stations(sts, st, date, db, collection, ip, port  ):
  # ищем станции рядом 
  query = {"st": { "$in": sts + [st] } }

  res      = get_query( db, collection, ip, port, query )
  count    = 0
  stations = []
  station  = []
  # проходимся по ответу
  for doc in res:
    # если станция уже есть - не добавляем
    if ( doc['st']==st and station==[] ):
      station=doc
    elif doc not in stations:
      stations.append(doc)

  # itemlist = [ item.get_byparam(param) for item in itemarray.get_all() ]
  v= [ [s['name'],s['st']] for s in stations]
  print(v)
  v= [ station['name'],station['st'], station['loc'] ]
  print(v)
  # возвращаем станции
  return stations, station




# 
# 
# ==============   POSTGRESQL   ================
# 
# 



def read_postgres( table=None, where=None, limit=None, query=None, sort=None, offset=None,
                   raw_query=None,
                   host='10.10.10.8',nmspace="public",db='db_meteo',user='postgres',password='' ):
  """Функция запроса в БД постгрес
  
  """
  cursor = connect_postgres( host=host,db=db,user=user,password=password )

  if where is None:
      where=''
  # if limit is None:
      # limit='100'
  if query is None:
    query=''
  if sort is None:
    sort=''

  if raw_query is None:
    cursor.execute('SELECT '+('*' if query=='' else query )+' FROM "'+nmspace+'".'+table+' '+( 'WHERE '+where if where!='' else '' )+' '+( 'ORDER BY '+sort if sort!='' else '' )+( '' if limit is None else ' LIMIT '+limit )+( '' if offset is None else ' OFFSET '+offset ) )
    print('SELECT '+('*' if query=='' else query )+' FROM "'+nmspace+'".'+table+' '+( 'WHERE '+where if where!='' else '' )+' '+( 'ORDER BY '+sort if sort!='' else '' )+( '' if limit is None else ' LIMIT '+limit )+( '' if offset is None else ' OFFSET '+offset ))
  else:
    cursor.execute(raw_query)
    print(raw_query)

  # df = pd.DataFrame(list(cursor))
  return cursor

# 
# подключаемся к БД
# 
def connect_postgres(host='10.10.10.7',db='clidb_gis',user='postgres',password=''):
  # строка подключения
  connect_str = "host='"+host+"' dbname='"+db+"' user='"+user+"' password='"+password+"'";
  # print(connect_str)
  # соединяемся с базой
  connect = pg2.connect(connect_str)
  # получаем курсор - указатель
  cursor = connect.cursor()

  return cursor

